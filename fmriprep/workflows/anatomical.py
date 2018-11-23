# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Anatomical reference preprocessing workflows
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_anat_preproc_wf
.. autofunction:: init_skullstrip_ants_wf

Surface preprocessing
+++++++++++++++++++++

``fmriprep`` uses FreeSurfer_ to reconstruct surfaces from T1w/T2w
structural images.

.. autofunction:: init_surface_recon_wf
.. autofunction:: init_autorecon_resume_wf
.. autofunction:: init_gifti_surface_wf

"""

import os.path as op

from pkg_resources import resource_filename as pkgr

from nipype.pipeline import engine as pe
from nipype.interfaces import (
    io as nio,
    utility as niu,
    c3,
    freesurfer as fs,
    fsl,
    image,
)
from nipype.interfaces.ants import BrainExtraction, N4BiasFieldCorrection

from niworkflows.interfaces.registration import RobustMNINormalizationRPT
import niworkflows.data as nid
from niworkflows.interfaces.masks import ROIsPlot

from niworkflows.interfaces.segmentation import ReconAllRPT
from niworkflows.interfaces.fixes import FixHeaderApplyTransforms as ApplyTransforms

from ..engine import Workflow
from ..interfaces import (
    DerivativesDataSink, MakeMidthickness, FSInjectBrainExtracted,
    FSDetectInputs, NormalizeSurf, GiftiNameSource, TemplateDimensions, Conform,
    ConcatAffines, RefineBrainMask,
)
from ..utils.misc import fix_multi_T1w_source_name, add_suffix


TEMPLATE_MAP = {
    'MNI152NLin2009cAsym': 'mni_icbm152_nlin_asym_09c',
    }


# #  pylint: disable=R0914
def init_anat_preproc_wf(skull_strip_template, output_spaces, template, debug,
                         freesurfer, longitudinal, omp_nthreads, hires, reportlets_dir,
                         output_dir, num_t1w,
                         skull_strip_fixed_seed=False, name='anat_preproc_wf'):
    """
    This workflow controls the anatomical preprocessing stages of FMRIPREP.

    This includes:

     - Creation of a structural template
     - Skull-stripping and bias correction
     - Tissue segmentation
     - Normalization
     - Surface reconstruction with FreeSurfer

    .. workflow::
        :graph2use: orig
        :simple_form: yes

        from fmriprep.workflows.anatomical import init_anat_preproc_wf
        wf = init_anat_preproc_wf(omp_nthreads=1,
                                  reportlets_dir='.',
                                  output_dir='.',
                                  template='MNI152NLin2009cAsym',
                                  output_spaces=['T1w', 'fsnative',
                                                 'template', 'fsaverage5'],
                                  skull_strip_template='OASIS',
                                  freesurfer=True,
                                  longitudinal=False,
                                  debug=False,
                                  hires=True,
                                  num_t1w=1)

    **Parameters**

        skull_strip_template : str
            Name of ANTs skull-stripping template ('OASIS' or 'NKI')
        output_spaces : list
            List of output spaces functional images are to be resampled to.

            Some pipeline components will only be instantiated for some output spaces.

            Valid spaces:

              - T1w
              - template
              - fsnative
              - fsaverage (or other pre-existing FreeSurfer templates)
        template : str
            Name of template targeted by ``template`` output space
        debug : bool
            Enable debugging outputs
        freesurfer : bool
            Enable FreeSurfer surface reconstruction (may increase runtime)
        longitudinal : bool
            Create unbiased structural template, regardless of number of inputs
            (may increase runtime)
        omp_nthreads : int
            Maximum number of threads an individual process may use
        hires : bool
            Enable sub-millimeter preprocessing in FreeSurfer
        reportlets_dir : str
            Directory in which to save reportlets
        output_dir : str
            Directory in which to save derivatives
        name : str, optional
            Workflow name (default: anat_preproc_wf)
        skull_strip_fixed_seed : bool
            Do not use a random seed for skull-stripping - will ensure
            run-to-run replicability when used with --omp-nthreads 1 (default: ``False``)


    **Inputs**

        t1w
            List of T1-weighted structural images
        t2w
            List of T2-weighted structural images
        flair
            List of FLAIR images
        subjects_dir
            FreeSurfer SUBJECTS_DIR


    **Outputs**

        t1_preproc
            Bias-corrected structural template, defining T1w space
        t1_brain
            Skull-stripped ``t1_preproc``
        t1_mask
            Mask of the skull-stripped template image
        t1_seg
            Segmentation of preprocessed structural image, including
            gray-matter (GM), white-matter (WM) and cerebrospinal fluid (CSF)
        t1_tpms
            List of tissue probability maps in T1w space
        t1_2_mni
            T1w template, normalized to MNI space
        t1_2_mni_forward_transform
            ANTs-compatible affine-and-warp transform file
        t1_2_mni_reverse_transform
            ANTs-compatible affine-and-warp transform file (inverse)
        mni_mask
            Mask of skull-stripped template, in MNI space
        mni_seg
            Segmentation, resampled into MNI space
        mni_tpms
            List of tissue probability maps in MNI space
        subjects_dir
            FreeSurfer SUBJECTS_DIR
        subject_id
            FreeSurfer subject ID
        t1_2_fsnative_forward_transform
            LTA-style affine matrix translating from T1w to FreeSurfer-conformed subject space
        t1_2_fsnative_reverse_transform
            LTA-style affine matrix translating from FreeSurfer-conformed subject space to T1w
        surfaces
            GIFTI surfaces (gray/white boundary, midthickness, pial, inflated)

    **Subworkflows**

        * :py:func:`~fmriprep.workflows.anatomical.init_skullstrip_ants_wf`
        * :py:func:`~fmriprep.workflows.anatomical.init_surface_recon_wf`

    """

    workflow = Workflow(name=name)
    workflow.__postdesc__ = """\
Spatial normalization to the ICBM 152 Nonlinear Asymmetrical
template version 2009c [@mni, RRID:SCR_008796] was performed
through nonlinear registration with `antsRegistration`
[ANTs {ants_ver}, RRID:SCR_004757, @ants], using
brain-extracted versions of both T1w volume and template.
Brain tissue segmentation of cerebrospinal fluid (CSF),
white-matter (WM) and gray-matter (GM) was performed on
the brain-extracted T1w using `fast` [FSL {fsl_ver}, RRID:SCR_002823,
@fsl_fast].
""".format(
        ants_ver=BrainExtraction().version or '<ver>',
        fsl_ver=fsl.FAST().version or '<ver>',
    )
    desc = """Anatomical data preprocessing

: """
    desc += """\
A total of {num_t1w} T1-weighted (T1w) images were found within the input
BIDS dataset.
All of them were corrected for intensity non-uniformity (INU)
using `N4BiasFieldCorrection` [@n4, ANTs {ants_ver}].
""" if num_t1w > 1 else """\
The T1-weighted (T1w) image was corrected for intensity non-uniformity (INU)
using `N4BiasFieldCorrection` [@n4, ANTs {ants_ver}],
and used as T1w-reference throughout the workflow.
"""

    workflow.__desc__ = desc.format(
        num_t1w=num_t1w,
        ants_ver=BrainExtraction().version or '<ver>'
    )

    ### ORIGINAL
    # inputnode = pe.Node(
    #     niu.IdentityInterface(fields=['t1w', 't2w', 'roi', 'flair', 'subjects_dir', 'subject_id']),
    #     name='inputnode')
    ### ADJUSTED SM (added inv2, t1map for MP2RAGE)
    inputnode = pe.Node(
        niu.IdentityInterface(fields=['t1w', 'inv2', 't1map', 't2w', 'roi', 'flair', 'subjects_dir', 'subject_id']),
        name='inputnode')

    buffernode = pe.Node(niu.IdentityInterface(
        fields=['t1_brain', 't1_mask']), name='buffernode')
    ### END ADJUSTED

    outputnode = pe.Node(niu.IdentityInterface(
        fields=['t1_preproc', 't1_brain', 't1_mask', 't1_seg', 't1_tpms',
                't1_2_mni', 't1_2_mni_forward_transform', 't1_2_mni_reverse_transform',
                'mni_mask', 'mni_seg', 'mni_tpms',
                'template_transforms',
                'subjects_dir', 'subject_id', 't1_2_fsnative_forward_transform',
                't1_2_fsnative_reverse_transform', 'surfaces', 't1_aseg', 't1_aparc']),
        name='outputnode')
    anat_template_wf = init_anat_template_wf(longitudinal=longitudinal, omp_nthreads=omp_nthreads,
                                             num_t1w=num_t1w)

    # 3. Skull-stripping
    ### ORIGINAL
    # Bias field correction is handled in skull strip workflows.
    # skullstrip_ants_wf = init_skullstrip_ants_wf(name='skullstrip_ants_wf',
    #                                              skull_strip_template=skull_strip_template,
    #                                              debug=debug,
    #                                              omp_nthreads=omp_nthreads)
    # workflow.connect([
    #     (inputnode, anat_template_wf, [('t1w', 'inputnode.t1w')]),
    #     (anat_template_wf, skullstrip_ants_wf, [('outputnode.t1_template', 'inputnode.in_file')]),
    #     (skullstrip_ants_wf, outputnode, [('outputnode.bias_corrected', 't1_preproc')]),
    #     (anat_template_wf, outputnode, [
    #         ('outputnode.template_transforms', 't1_template_transforms')]),
    #     (buffernode, outputnode, [('t1_brain', 't1_brain'),
    #                               ('t1_mask', 't1_mask')]),
    # ])
    ### ADJUSTED SM (added custom skull-stripping that works for MP2RAGE, based on nighres)
    skull_stripper = pe.Node(niu.Function(input_names=['second_inversion', 't1_weighted', 't1_map'],
                                          output_names=['brain_mask', 'inv2_masked', 't1w_masked', 't1map_masked'],
                                          function=skull_strip), name='skstrp')
    workflow.connect([
        (inputnode, anat_template_wf, [('t1w', 'inputnode.t1w')]),
        (inputnode, skull_stripper, [(('t1w', get_first), 't1_weighted'),
                                     ('inv2', 'second_inversion'),
                                     (('t1map', get_first), 't1_map')]),
        (skull_stripper, outputnode, [('t1w_masked', 't1_preproc')]),
        (anat_template_wf, outputnode, [
            ('outputnode.template_transforms', 't1_template_transforms')]),
        (buffernode, outputnode, [('t1_brain', 't1_brain'),
                                  ('t1_mask', 't1_mask')]),
    ])
    ### END ADJUSTED


    # 4. Surface reconstruction
    ### ORIGINAL
    # if freesurfer:
    #     surface_recon_wf = init_surface_recon_wf(name='surface_recon_wf',
    #                                              omp_nthreads=omp_nthreads, hires=hires)
    #     applyrefined = pe.Node(fsl.ApplyMask(), name='applyrefined')
    #     workflow.connect([
    #         (inputnode, surface_recon_wf, [
    #             ('t2w', 'inputnode.t2w'),
    #             ('flair', 'inputnode.flair'),
    #             ('subjects_dir', 'inputnode.subjects_dir'),
    #             ('subject_id', 'inputnode.subject_id')]),
    #         (anat_template_wf, surface_recon_wf, [('outputnode.t1_template', 'inputnode.t1w')]),
    #         (skullstrip_ants_wf, surface_recon_wf, [
    #             ('outputnode.out_file', 'inputnode.skullstripped_t1'),
    #             ('outputnode.out_segs', 'inputnode.ants_segs'),
    #             ('outputnode.bias_corrected', 'inputnode.corrected_t1')]),
    #         (skullstrip_ants_wf, applyrefined, [
    #             ('outputnode.bias_corrected', 'in_file')]),
    #         (surface_recon_wf, applyrefined, [
    #             ('outputnode.out_brainmask', 'mask_file')]),
    #         (surface_recon_wf, outputnode, [
    #             ('outputnode.subjects_dir', 'subjects_dir'),
    #             ('outputnode.subject_id', 'subject_id'),
    #             ('outputnode.t1_2_fsnative_forward_transform', 't1_2_fsnative_forward_transform'),
    #             ('outputnode.t1_2_fsnative_reverse_transform', 't1_2_fsnative_reverse_transform'),
    #             ('outputnode.surfaces', 'surfaces'),
    #             ('outputnode.out_aseg', 't1_aseg'),
    #             ('outputnode.out_aparc', 't1_aparc')]),
    #         (applyrefined, buffernode, [('out_file', 't1_brain')]),
    #         (surface_recon_wf, buffernode, [
    #             ('outputnode.out_brainmask', 't1_mask')]),
    #     ])
    # else:
    #     workflow.connect([
    #         (skullstrip_ants_wf, buffernode, [
    #             ('outputnode.out_file', 't1_brain'),
    #             ('outputnode.out_mask', 't1_mask')]),
    #     ])
    ### ADJUSTED SM (WARNING: untested; I don't use surface reconstruction). Adjustment connect customized
    # skull_stripper
    if freesurfer:
        surface_recon_wf = init_surface_recon_wf(name='surface_recon_wf',
                                                 omp_nthreads=omp_nthreads, hires=hires)
        applyrefined = pe.Node(fsl.ApplyMask(), name='applyrefined')
        workflow.connect([
            (inputnode, surface_recon_wf, [
                ('t2w', 'inputnode.t2w'),
                ('flair', 'inputnode.flair'),
                ('subjects_dir', 'inputnode.subjects_dir'),
                ('subject_id', 'inputnode.subject_id')]),
            (anat_template_wf, surface_recon_wf, [('outputnode.t1_template', 'inputnode.t1w')]),
            (skull_stripper, surface_recon_wf, [
                ('outputnode.out_file', 'inputnode.skullstripped_t1'),
                ('outputnode.out_segs', 'inputnode.ants_segs'),
                ('outputnode.bias_corrected', 'inputnode.corrected_t1')]),
            (skull_stripper, applyrefined, [
                ('outputnode.bias_corrected', 'in_file')]),
            (surface_recon_wf, applyrefined, [
                ('outputnode.out_brainmask', 'mask_file')]),
            (surface_recon_wf, outputnode, [
                ('outputnode.subjects_dir', 'subjects_dir'),
                ('outputnode.subject_id', 'subject_id'),
                ('outputnode.t1_2_fsnative_forward_transform', 't1_2_fsnative_forward_transform'),
                ('outputnode.t1_2_fsnative_reverse_transform', 't1_2_fsnative_reverse_transform'),
                ('outputnode.surfaces', 'surfaces'),
                ('outputnode.out_aseg', 't1_aseg'),
                ('outputnode.out_aparc', 't1_aparc')]),
            (applyrefined, buffernode, [('out_file', 't1_brain')]),
            (surface_recon_wf, buffernode, [
                ('outputnode.out_brainmask', 't1_mask')]),
        ])
    else:
        workflow.connect([
            (skull_stripper, buffernode, [
                ('t1w_masked', 't1_brain'),
                ('brain_mask', 't1_mask')]),
        ])
    ### END ADJUSTED

    # 5. Segmentation
    t1_seg = pe.Node(fsl.FAST(segments=True, no_bias=True, probability_maps=True),
                     name='t1_seg', mem_gb=3)

    workflow.connect([
        (buffernode, t1_seg, [('t1_brain', 'in_files')]),
        (t1_seg, outputnode, [('tissue_class_map', 't1_seg'),
                              ('probability_maps', 't1_tpms')]),
    ])

    # 6. Spatial normalization (T1w to MNI registration)
    ### ORIGINAL
    # t1_2_mni = pe.Node(
    #     RobustMNINormalizationRPT(
    #         float=True,
    #         generate_report=True,
    #         flavor='testing' if debug else 'precise',
    #     ),
    #     name='t1_2_mni',
    #     n_procs=omp_nthreads,
    #     mem_gb=2
    # )
    ### ADJUSTED SM (use nighres embedded_ants registration routine)
    t1_2_mni = pe.Node(niu.Function(input_names=['source_img', 'target_img'],
                                    output_names=['transformed_source', 'mapping', 'inverse',
                                                  'mapping_cmap', 'inverse_cmap', 'out_report'],
                                    function=register_func),
                       name='calculate_warp',
                       mem_gb=15)  # educated guess based on ~12% usage on carcajou
    ### END ADJUSTED

    # Resample the brain mask and the tissue probability maps into mni space
    mni_mask = pe.Node(
        ApplyTransforms(dimension=3, default_value=0, float=True,
                        interpolation='MultiLabel'),
        name='mni_mask'
    )

    mni_seg = pe.Node(
        ApplyTransforms(dimension=3, default_value=0, float=True,
                        interpolation='MultiLabel'),
        name='mni_seg'
    )

    mni_tpms = pe.MapNode(
        ApplyTransforms(dimension=3, default_value=0, float=True,
                        interpolation='Linear'),
        iterfield=['input_image'],
        name='mni_tpms'
    )

    if 'template' in output_spaces:
        ### ORIGINAL
        # template_str = TEMPLATE_MAP[template]
        # ref_img = op.join(nid.get_dataset(template_str), '1mm_T1.nii.gz')
        #
        # t1_2_mni.inputs.template = template_str
        ### ADJUSTED SM (allow for custom template - we assume it's stored in /data/template).
        ref_img = '/data/templates/' + template + '.nii.gz'
        t1_2_mni.inputs.target_img = ref_img
        ### END ADJUSTED

        mni_mask.inputs.reference_image = ref_img
        mni_seg.inputs.reference_image = ref_img
        mni_tpms.inputs.reference_image = ref_img

        workflow.connect([
            (inputnode, t1_2_mni, [('roi', 'lesion_mask')]),
            ### ADJUSTED SM (customized registration requires different inputs)
            # (skullstrip_ants_wf, t1_2_mni, [('outputnode.bias_corrected', 'moving_image')]),
            # (buffernode, t1_2_mni, [('t1_mask', 'moving_mask')]),
            (skull_stripper, t1_2_mni, [('t1w_masked', 'source_img')]),
            ### END ADJUSTED
            (buffernode, mni_mask, [('t1_mask', 'input_image')]),
            (t1_2_mni, mni_mask, [('mapping', 'transforms')]),  # SM: adjusted output name
            (t1_seg, mni_seg, [('tissue_class_map', 'input_image')]),
            (t1_2_mni, mni_seg, [('mapping', 'transforms')]),  # SM: adjusted output name
            (t1_seg, mni_tpms, [('probability_maps', 'input_image')]),
            (t1_2_mni, mni_tpms, [('mapping', 'transforms')]),  # SM: adjusted output name
            (t1_2_mni, outputnode, [
                ('transformed_source', 't1_2_mni'),  # SM: adjusted output name
                ('mapping', 't1_2_mni_forward_transform'),  # SM: adjusted output name
                ('inverse', 't1_2_mni_reverse_transform'),
                ('mapping_cmap', 't1_2_mni_forward_transform_cmap'),
                ('inverse_cmap', 't1_2_mni_reverse_transform_cmap')]),  # SM: adjusted output name
            (mni_mask, outputnode, [('output_image', 'mni_mask')]),
            (mni_seg, outputnode, [('output_image', 'mni_seg')]),
            (mni_tpms, outputnode, [('output_image', 'mni_tpms')]),
        ])

    seg2msks = pe.Node(niu.Function(function=_seg2msks), name='seg2msks')
    seg_rpt = pe.Node(ROIsPlot(colors=['r', 'magenta', 'b', 'g']), name='seg_rpt')
    anat_reports_wf = init_anat_reports_wf(
        reportlets_dir=reportlets_dir, output_spaces=output_spaces, template=template,
        freesurfer=freesurfer)
    workflow.connect([
        (inputnode, anat_reports_wf, [
            (('t1w', fix_multi_T1w_source_name), 'inputnode.source_file')]),
        (anat_template_wf, anat_reports_wf, [
            ('outputnode.out_report', 'inputnode.t1_conform_report')]),
        (anat_template_wf, seg_rpt, [
            ('outputnode.t1_template', 'in_file')]),
        (t1_seg, seg2msks, [('tissue_class_map', 'in_file')]),
        (seg2msks, seg_rpt, [('out', 'in_rois')]),
        (outputnode, seg_rpt, [('t1_mask', 'in_mask')]),
        (seg_rpt, anat_reports_wf, [('out_report', 'inputnode.seg_report')]),
    ])

    if freesurfer:
        workflow.connect([
            (surface_recon_wf, anat_reports_wf, [
                ('outputnode.out_report', 'inputnode.recon_report')]),
        ])
    if 'template' in output_spaces:
        workflow.connect([
            (t1_2_mni, anat_reports_wf, [('out_report', 'inputnode.t1_2_mni_report')]),
        ])

    anat_derivatives_wf = init_anat_derivatives_wf(output_dir=output_dir,
                                                   output_spaces=output_spaces,
                                                   template=template,
                                                   freesurfer=freesurfer)

    workflow.connect([
        (anat_template_wf, anat_derivatives_wf, [
            ('outputnode.t1w_valid_list', 'inputnode.source_files')]),
        (outputnode, anat_derivatives_wf, [
            ('t1_template_transforms', 'inputnode.t1_template_transforms'),
            ('t1_preproc', 'inputnode.t1_preproc'),
            ('t1_mask', 'inputnode.t1_mask'),
            ('t1_seg', 'inputnode.t1_seg'),
            ('t1_tpms', 'inputnode.t1_tpms'),
            ('t1_2_mni_forward_transform', 'inputnode.t1_2_mni_forward_transform'),
            ('t1_2_mni_forward_transform_cmap', 'inputnode.t1_2_mni_forward_transform_cmap'),  # SM: add coord map
            ('t1_2_mni_reverse_transform', 'inputnode.t1_2_mni_reverse_transform'),
            ('t1_2_mni_reverse_transform_cmap', 'inputnode.t1_2_mni_reverse_transform_cmap'),  # SM: add coord map
            ('t1_2_mni', 'inputnode.t1_2_mni'),
            ('mni_mask', 'inputnode.mni_mask'),
            ('mni_seg', 'inputnode.mni_seg'),
            ('mni_tpms', 'inputnode.mni_tpms'),
            ('t1_2_fsnative_forward_transform', 'inputnode.t1_2_fsnative_forward_transform'),
            ('surfaces', 'inputnode.surfaces'),
        ]),
    ])

    if freesurfer:
        workflow.connect([
            (surface_recon_wf, anat_derivatives_wf, [
                ('outputnode.out_aseg', 'inputnode.t1_fs_aseg'),
                ('outputnode.out_aparc', 'inputnode.t1_fs_aparc'),
            ]),
        ])

    return workflow


def init_anat_template_wf(longitudinal, omp_nthreads, num_t1w, name='anat_template_wf'):
    r"""
    This workflow generates a canonically oriented structural template from
    input T1w images.


    .. workflow::
        :graph2use: orig
        :simple_form: yes

        from fmriprep.workflows.anatomical import init_anat_template_wf
        wf = init_anat_template_wf(longitudinal=False, omp_nthreads=1, num_t1w=1)

    **Parameters**

        longitudinal : bool
            Create unbiased structural template, regardless of number of inputs
            (may increase runtime)
        omp_nthreads : int
            Maximum number of threads an individual process may use
        num_t1w : int
            Number of T1w images
        name : str, optional
            Workflow name (default: anat_template_wf)


    **Inputs**

        t1w
            List of T1-weighted structural images


    **Outputs**

        t1_template
            Structural template, defining T1w space
        template_transforms
            List of affine transforms from ``t1_template`` to original T1w images
        out_report
            Conformation report
    """

    workflow = Workflow(name=name)

    if num_t1w > 1:
        workflow.__desc__ = """\
A T1w-reference map was computed after registration of
{num_t1w} T1w images (after INU-correction) using
`mri_robust_template` [FreeSurfer {fs_ver}, @fs_template].
""".format(num_t1w=num_t1w, fs_ver=fs.Info().looseversion() or '<ver>')

    inputnode = pe.Node(niu.IdentityInterface(fields=['t1w']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['t1_template', 't1w_valid_list', 'template_transforms', 'out_report']),
        name='outputnode')

    # 0. Reorient T1w image(s) to RAS and resample to common voxel space
    t1_template_dimensions = pe.Node(TemplateDimensions(), name='t1_template_dimensions')
    t1_conform = pe.MapNode(Conform(), iterfield='in_file', name='t1_conform')

    workflow.connect([
        (inputnode, t1_template_dimensions, [('t1w', 't1w_list')]),
        (t1_template_dimensions, t1_conform, [
            ('t1w_valid_list', 'in_file'),
            ('target_zooms', 'target_zooms'),
            ('target_shape', 'target_shape')]),
        (t1_template_dimensions, outputnode, [('out_report', 'out_report'),
                                              ('t1w_valid_list', 't1w_valid_list')]),
    ])

    if num_t1w == 1:
        def _get_first(in_list):
            if isinstance(in_list, (list, tuple)):
                return in_list[0]
            return in_list

        outputnode.inputs.template_transforms = [pkgr('fmriprep', 'data/itkIdentityTransform.txt')]

        workflow.connect([
            (t1_conform, outputnode, [(('out_file', _get_first), 't1_template')]),
        ])

        return workflow

    # 1. Template (only if several T1w images)
    # 1a. Correct for bias field: the bias field is an additive factor
    #     in log-transformed intensity units. Therefore, it is not a linear
    #     combination of fields and N4 fails with merged images.
    # 1b. Align and merge if several T1w images are provided
    n4_correct = pe.MapNode(
        N4BiasFieldCorrection(dimension=3, copy_header=True),
        iterfield='input_image', name='n4_correct',
        n_procs=1)  # n_procs=1 for reproducibility
    t1_merge = pe.Node(
        fs.RobustTemplate(auto_detect_sensitivity=True,
                          initial_timepoint=1,      # For deterministic behavior
                          intensity_scaling=True,   # 7-DOF (rigid + intensity)
                          subsample_threshold=200,
                          fixed_timepoint=not longitudinal,
                          no_iteration=not longitudinal,
                          transform_outputs=True,
                          ),
        mem_gb=2 * num_t1w - 1,
        name='t1_merge')

    # 2. Reorient template to RAS, if needed (mri_robust_template may set to LIA)
    t1_reorient = pe.Node(image.Reorient(), name='t1_reorient')

    lta_to_fsl = pe.MapNode(fs.utils.LTAConvert(out_fsl=True), iterfield=['in_lta'],
                            name='lta_to_fsl')

    concat_affines = pe.MapNode(
        ConcatAffines(3, invert=True), iterfield=['mat_AtoB', 'mat_BtoC'],
        name='concat_affines', run_without_submitting=True)

    fsl_to_itk = pe.MapNode(c3.C3dAffineTool(fsl2ras=True, itk_transform=True),
                            iterfield=['transform_file', 'source_file'], name='fsl_to_itk')

    def _set_threads(in_list, maximum):
        return min(len(in_list), maximum)

    workflow.connect([
        (t1_conform, n4_correct, [('out_file', 'input_image')]),
        (t1_conform, t1_merge, [
            (('out_file', _set_threads, omp_nthreads), 'num_threads'),
            (('out_file', add_suffix, '_template'), 'out_file')]),
        (n4_correct, t1_merge, [('output_image', 'in_files')]),
        (t1_merge, t1_reorient, [('out_file', 'in_file')]),
        # Combine orientation and template transforms
        (t1_merge, lta_to_fsl, [('transform_outputs', 'in_lta')]),
        (t1_conform, concat_affines, [('transform', 'mat_AtoB')]),
        (lta_to_fsl, concat_affines, [('out_fsl', 'mat_BtoC')]),
        (t1_reorient, concat_affines, [('transform', 'mat_CtoD')]),
        (t1_template_dimensions, fsl_to_itk, [('t1w_valid_list', 'source_file')]),
        (t1_reorient, fsl_to_itk, [('out_file', 'reference_file')]),
        (concat_affines, fsl_to_itk, [('out_mat', 'transform_file')]),
        # Output
        (t1_reorient, outputnode, [('out_file', 't1_template')]),
        (fsl_to_itk, outputnode, [('itk_transform', 'template_transforms')]),
    ])

    return workflow


def init_skullstrip_ants_wf(skull_strip_template, debug, omp_nthreads,
                            skull_strip_fixed_seed=False, name='skullstrip_ants_wf'):
    r"""
    This workflow performs skull-stripping using ANTs' ``BrainExtraction.sh``

    .. workflow::
        :graph2use: orig
        :simple_form: yes

        from fmriprep.workflows.anatomical import init_skullstrip_ants_wf
        wf = init_skullstrip_ants_wf(skull_strip_template='OASIS', debug=False, omp_nthreads=1)

    **Parameters**

        skull_strip_template : str
            Name of ANTs skull-stripping template ('OASIS' or 'NKI')
        debug : bool
            Enable debugging outputs
        omp_nthreads : int
            Maximum number of threads an individual process may use
        skull_strip_fixed_seed : bool
            Do not use a random seed for skull-stripping - will ensure
            run-to-run replicability when used with --omp-nthreads 1 (default: ``False``)

    **Inputs**

        in_file
            T1-weighted structural image to skull-strip

    **Outputs**

        bias_corrected
            Bias-corrected ``in_file``, before skull-stripping
        out_file
            Skull-stripped ``in_file``
        out_mask
            Binary mask of the skull-stripped ``in_file``
        out_report
            Reportlet visualizing quality of skull-stripping

    """
    from niworkflows.data.getters import get_dataset

    if skull_strip_template not in ['OASIS', 'NKI']:
        raise ValueError("Unknown skull-stripping template; select from {OASIS, NKI}")

    workflow = Workflow(name=name)
    workflow.__desc__ = """\
The T1w-reference was then skull-stripped using `antsBrainExtraction.sh`
(ANTs {ants_ver}), using {skullstrip_tpl} as target template.
""".format(ants_ver=BrainExtraction().version or '<ver>', skullstrip_tpl=skull_strip_template)

    # Grabbing the appropriate template elements
    template_dir = get_dataset('ants_%s_template_ras' % skull_strip_template.lower())
    brain_probability_mask = op.join(
        template_dir, 'T_template0_BrainCerebellumProbabilityMask.nii.gz')

    # TODO: normalize these names so this is not necessary
    if skull_strip_template == 'OASIS':
        brain_template = op.join(template_dir, 'T_template0.nii.gz')
        extraction_registration_mask = op.join(
            template_dir, 'T_template0_BrainCerebellumRegistrationMask.nii.gz')
    elif skull_strip_template == 'NKI':
        brain_template = op.join(template_dir, 'T_template.nii.gz')
        extraction_registration_mask = op.join(
            template_dir, 'T_template_BrainCerebellumExtractionMask.nii.gz')

    inputnode = pe.Node(niu.IdentityInterface(fields=['in_file']),
                        name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['bias_corrected', 'out_file', 'out_mask', 'out_segs', 'out_report']),
        name='outputnode')

    t1_skull_strip = pe.Node(
        BrainExtraction(dimension=3, use_floatingpoint_precision=1, debug=debug,
                        keep_temporary_files=1, use_random_seeding=not skull_strip_fixed_seed),
        name='t1_skull_strip', n_procs=omp_nthreads)

    t1_skull_strip.inputs.brain_template = brain_template
    t1_skull_strip.inputs.brain_probability_mask = brain_probability_mask
    t1_skull_strip.inputs.extraction_registration_mask = extraction_registration_mask

    workflow.connect([
        (inputnode, t1_skull_strip, [('in_file', 'anatomical_image')]),
        (t1_skull_strip, outputnode, [('BrainExtractionMask', 'out_mask'),
                                      ('BrainExtractionBrain', 'out_file'),
                                      ('BrainExtractionSegmentation', 'out_segs'),
                                      ('N4Corrected0', 'bias_corrected')])
    ])

    return workflow


def init_surface_recon_wf(omp_nthreads, hires, name='surface_recon_wf'):
    r"""
    This workflow reconstructs anatomical surfaces using FreeSurfer's ``recon-all``.

    Reconstruction is performed in three phases.
    The first phase initializes the subject with T1w and T2w (if available)
    structural images and performs basic reconstruction (``autorecon1``) with the
    exception of skull-stripping.
    For example, a subject with only one session with T1w and T2w images
    would be processed by the following command::

        $ recon-all -sd <output dir>/freesurfer -subjid sub-<subject_label> \
            -i <bids-root>/sub-<subject_label>/anat/sub-<subject_label>_T1w.nii.gz \
            -T2 <bids-root>/sub-<subject_label>/anat/sub-<subject_label>_T2w.nii.gz \
            -autorecon1 \
            -noskullstrip

    The second phase imports an externally computed skull-stripping mask.
    This workflow refines the external brainmask using the internal mask
    implicit the the FreeSurfer's ``aseg.mgz`` segmentation,
    to reconcile ANTs' and FreeSurfer's brain masks.

    First, the ``aseg.mgz`` mask from FreeSurfer is refined in two
    steps, using binary morphological operations:

      1. With a binary closing operation the sulci are included
         into the mask. This results in a smoother brain mask
         that does not exclude deep, wide sulci.

      2. Fill any holes (typically, there could be a hole next to
         the pineal gland and the corpora quadrigemina if the great
         cerebral brain is segmented out).

    Second, the brain mask is grown, including pixels that have a high likelihood
    to the GM tissue distribution:

      3. Dilate and substract the brain mask, defining the region to search for candidate
         pixels that likely belong to cortical GM.

      4. Pixels found in the search region that are labeled as GM by ANTs
         (during ``antsBrainExtraction.sh``) are directly added to the new mask.

      5. Otherwise, estimate GM tissue parameters locally in  patches of ``ww`` size,
         and test the likelihood of the pixel to belong in the GM distribution.

    This procedure is inspired on mindboggle's solution to the problem:
    https://github.com/nipy/mindboggle/blob/7f91faaa7664d820fe12ccc52ebaf21d679795e2/mindboggle/guts/segment.py#L1660


    The final phase resumes reconstruction, using the T2w image to assist
    in finding the pial surface, if available.
    See :py:func:`~fmriprep.workflows.anatomical.init_autorecon_resume_wf` for details.


    Memory annotations for FreeSurfer are based off `their documentation
    <https://surfer.nmr.mgh.harvard.edu/fswiki/SystemRequirements>`_.
    They specify an allocation of 4GB per subject. Here we define 5GB
    to have a certain margin.



    .. workflow::
        :graph2use: orig
        :simple_form: yes

        from fmriprep.workflows.anatomical import init_surface_recon_wf
        wf = init_surface_recon_wf(omp_nthreads=1, hires=True)

    **Parameters**

        omp_nthreads : int
            Maximum number of threads an individual process may use
        hires : bool
            Enable sub-millimeter preprocessing in FreeSurfer

    **Inputs**

        t1w
            List of T1-weighted structural images
        t2w
            List of T2-weighted structural images (only first used)
        flair
            List of FLAIR images
        skullstripped_t1
            Skull-stripped T1-weighted image (or mask of image)
        ants_segs
            Brain tissue segmentation from ANTS ``antsBrainExtraction.sh``
        corrected_t1
            INU-corrected, merged T1-weighted image
        subjects_dir
            FreeSurfer SUBJECTS_DIR
        subject_id
            FreeSurfer subject ID

    **Outputs**

        subjects_dir
            FreeSurfer SUBJECTS_DIR
        subject_id
            FreeSurfer subject ID
        t1_2_fsnative_forward_transform
            LTA-style affine matrix translating from T1w to FreeSurfer-conformed subject space
        t1_2_fsnative_reverse_transform
            LTA-style affine matrix translating from FreeSurfer-conformed subject space to T1w
        surfaces
            GIFTI surfaces for gray/white matter boundary, pial surface,
            midthickness (or graymid) surface, and inflated surfaces
        out_brainmask
            Refined brainmask, derived from FreeSurfer's ``aseg`` volume
        out_aseg
            FreeSurfer's aseg segmentation, in native T1w space
        out_aparc
            FreeSurfer's aparc+aseg segmentation, in native T1w space
        out_report
            Reportlet visualizing quality of surface alignment

    **Subworkflows**

        * :py:func:`~fmriprep.workflows.anatomical.init_autorecon_resume_wf`
        * :py:func:`~fmriprep.workflows.anatomical.init_gifti_surface_wf`
    """

    workflow = Workflow(name=name)
    workflow.__desc__ = """\
Brain surfaces were reconstructed using `recon-all` [FreeSurfer {fs_ver},
RRID:SCR_001847, @fs_reconall], and the brain mask estimated
previously was refined with a custom variation of the method to reconcile
ANTs-derived and FreeSurfer-derived segmentations of the cortical
gray-matter of Mindboggle [RRID:SCR_002438, @mindboggle].
""".format(fs_ver=fs.Info().looseversion() or '<ver>')

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=['t1w', 't2w', 'flair', 'skullstripped_t1', 'corrected_t1', 'ants_segs',
                    'subjects_dir', 'subject_id']), name='inputnode')
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=['subjects_dir', 'subject_id', 't1_2_fsnative_forward_transform',
                    't1_2_fsnative_reverse_transform', 'surfaces', 'out_brainmask',
                    'out_aseg', 'out_aparc', 'out_report']),
        name='outputnode')

    recon_config = pe.Node(FSDetectInputs(hires_enabled=hires), name='recon_config')

    autorecon1 = pe.Node(
        fs.ReconAll(directive='autorecon1', flags='-noskullstrip', openmp=omp_nthreads),
        name='autorecon1', n_procs=omp_nthreads, mem_gb=5)
    autorecon1.interface._can_resume = False

    skull_strip_extern = pe.Node(FSInjectBrainExtracted(), name='skull_strip_extern')

    fsnative_2_t1_xfm = pe.Node(fs.RobustRegister(auto_sens=True, est_int_scale=True),
                                name='fsnative_2_t1_xfm')
    t1_2_fsnative_xfm = pe.Node(fs.utils.LTAConvert(out_lta=True, invert=True),
                                name='t1_2_fsnative_xfm')

    autorecon_resume_wf = init_autorecon_resume_wf(omp_nthreads=omp_nthreads)
    gifti_surface_wf = init_gifti_surface_wf()

    aseg_to_native_wf = init_segs_to_native_wf()
    aparc_to_native_wf = init_segs_to_native_wf(segmentation='aparc_aseg')
    refine = pe.Node(RefineBrainMask(), name='refine')

    workflow.connect([
        # Configuration
        (inputnode, recon_config, [('t1w', 't1w_list'),
                                   ('t2w', 't2w_list'),
                                   ('flair', 'flair_list')]),
        # Passing subjects_dir / subject_id enforces serial order
        (inputnode, autorecon1, [('subjects_dir', 'subjects_dir'),
                                 ('subject_id', 'subject_id')]),
        (autorecon1, skull_strip_extern, [('subjects_dir', 'subjects_dir'),
                                          ('subject_id', 'subject_id')]),
        (skull_strip_extern, autorecon_resume_wf, [('subjects_dir', 'inputnode.subjects_dir'),
                                                   ('subject_id', 'inputnode.subject_id')]),
        (autorecon_resume_wf, gifti_surface_wf, [
            ('outputnode.subjects_dir', 'inputnode.subjects_dir'),
            ('outputnode.subject_id', 'inputnode.subject_id')]),
        # Reconstruction phases
        (inputnode, autorecon1, [('t1w', 'T1_files')]),
        (recon_config, autorecon1, [('t2w', 'T2_file'),
                                    ('flair', 'FLAIR_file'),
                                    ('hires', 'hires'),
                                    # First run only (recon-all saves expert options)
                                    ('mris_inflate', 'mris_inflate')]),
        (inputnode, skull_strip_extern, [('skullstripped_t1', 'in_brain')]),
        (recon_config, autorecon_resume_wf, [('use_t2w', 'inputnode.use_T2'),
                                             ('use_flair', 'inputnode.use_FLAIR')]),
        # Construct transform from FreeSurfer conformed image to FMRIPREP
        # reoriented image
        (inputnode, fsnative_2_t1_xfm, [('t1w', 'target_file')]),
        (autorecon1, fsnative_2_t1_xfm, [('T1', 'source_file')]),
        (fsnative_2_t1_xfm, gifti_surface_wf, [
            ('out_reg_file', 'inputnode.t1_2_fsnative_reverse_transform')]),
        (fsnative_2_t1_xfm, t1_2_fsnative_xfm, [('out_reg_file', 'in_lta')]),
        # Refine ANTs mask, deriving new mask from FS' aseg
        (inputnode, refine, [('corrected_t1', 'in_anat'),
                             ('ants_segs', 'in_ants')]),
        (inputnode, aseg_to_native_wf, [('corrected_t1', 'inputnode.in_file')]),
        (autorecon_resume_wf, aseg_to_native_wf, [
            ('outputnode.subjects_dir', 'inputnode.subjects_dir'),
            ('outputnode.subject_id', 'inputnode.subject_id')]),
        (inputnode, aparc_to_native_wf, [('corrected_t1', 'inputnode.in_file')]),
        (autorecon_resume_wf, aparc_to_native_wf, [
            ('outputnode.subjects_dir', 'inputnode.subjects_dir'),
            ('outputnode.subject_id', 'inputnode.subject_id')]),
        (aseg_to_native_wf, refine, [('outputnode.out_file', 'in_aseg')]),

        # Output
        (autorecon_resume_wf, outputnode, [('outputnode.subjects_dir', 'subjects_dir'),
                                           ('outputnode.subject_id', 'subject_id'),
                                           ('outputnode.out_report', 'out_report')]),
        (gifti_surface_wf, outputnode, [('outputnode.surfaces', 'surfaces')]),
        (t1_2_fsnative_xfm, outputnode, [('out_lta', 't1_2_fsnative_forward_transform')]),
        (fsnative_2_t1_xfm, outputnode, [('out_reg_file', 't1_2_fsnative_reverse_transform')]),
        (refine, outputnode, [('out_file', 'out_brainmask')]),
        (aseg_to_native_wf, outputnode, [('outputnode.out_file', 'out_aseg')]),
        (aparc_to_native_wf, outputnode, [('outputnode.out_file', 'out_aparc')]),
    ])

    return workflow


def init_autorecon_resume_wf(omp_nthreads, name='autorecon_resume_wf'):
    r"""
    This workflow resumes recon-all execution, assuming the `-autorecon1` stage
    has been completed.

    In order to utilize resources efficiently, this is broken down into five
    sub-stages; after the first stage, the second and third stages may be run
    simultaneously, and the fourth and fifth stages may be run simultaneously,
    if resources permit::

        $ recon-all -sd <output dir>/freesurfer -subjid sub-<subject_label> \
            -autorecon2-volonly
        $ recon-all -sd <output dir>/freesurfer -subjid sub-<subject_label> \
            -autorecon-hemi lh \
            -noparcstats -nocortparc2 -noparcstats2 -nocortparc3 \
            -noparcstats3 -nopctsurfcon -nohyporelabel -noaparc2aseg \
            -noapas2aseg -nosegstats -nowmparc -nobalabels
        $ recon-all -sd <output dir>/freesurfer -subjid sub-<subject_label> \
            -autorecon-hemi rh \
            -noparcstats -nocortparc2 -noparcstats2 -nocortparc3 \
            -noparcstats3 -nopctsurfcon -nohyporelabel -noaparc2aseg \
            -noapas2aseg -nosegstats -nowmparc -nobalabels
        $ recon-all -sd <output dir>/freesurfer -subjid sub-<subject_label> \
            -autorecon3 -hemi lh -T2pial
        $ recon-all -sd <output dir>/freesurfer -subjid sub-<subject_label> \
            -autorecon3 -hemi rh -T2pial

    The excluded steps in the second and third stages (``-no<option>``) are not
    fully hemisphere independent, and are therefore postponed to the final two
    stages.

    .. workflow::
        :graph2use: orig
        :simple_form: yes

        from fmriprep.workflows.anatomical import init_autorecon_resume_wf
        wf = init_autorecon_resume_wf(omp_nthreads=1)

    **Inputs**

        subjects_dir
            FreeSurfer SUBJECTS_DIR
        subject_id
            FreeSurfer subject ID
        use_T2
            Refine pial surface using T2w image
        use_FLAIR
            Refine pial surface using FLAIR image

    **Outputs**

        subjects_dir
            FreeSurfer SUBJECTS_DIR
        subject_id
            FreeSurfer subject ID
        out_report
            Reportlet visualizing quality of surface alignment

    """
    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=['subjects_dir', 'subject_id', 'use_T2', 'use_FLAIR']),
        name='inputnode')

    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=['subjects_dir', 'subject_id', 'out_report']),
        name='outputnode')

    autorecon2_vol = pe.Node(
        fs.ReconAll(directive='autorecon2-volonly', openmp=omp_nthreads),
        n_procs=omp_nthreads, mem_gb=5, name='autorecon2_vol')

    autorecon_surfs = pe.MapNode(
        fs.ReconAll(
            directive='autorecon-hemi',
            flags=['-noparcstats', '-nocortparc2', '-noparcstats2',
                   '-nocortparc3', '-noparcstats3', '-nopctsurfcon',
                   '-nohyporelabel', '-noaparc2aseg', '-noapas2aseg',
                   '-nosegstats', '-nowmparc', '-nobalabels'],
            openmp=omp_nthreads),
        iterfield='hemi', n_procs=omp_nthreads, mem_gb=5,
        name='autorecon_surfs')
    autorecon_surfs.inputs.hemi = ['lh', 'rh']

    autorecon3 = pe.MapNode(
        fs.ReconAll(directive='autorecon3', openmp=omp_nthreads),
        iterfield='hemi', n_procs=omp_nthreads, mem_gb=5,
        name='autorecon3')
    autorecon3.inputs.hemi = ['lh', 'rh']

    # Only generate the report once; should be nothing to do
    recon_report = pe.Node(
        ReconAllRPT(directive='autorecon3', generate_report=True),
        name='recon_report', mem_gb=5)

    def _dedup(in_list):
        vals = set(in_list)
        if len(vals) > 1:
            raise ValueError(
                "Non-identical values can't be deduplicated:\n{!r}".format(in_list))
        return vals.pop()

    workflow.connect([
        (inputnode, autorecon3, [('use_T2', 'use_T2'),
                                 ('use_FLAIR', 'use_FLAIR')]),
        (inputnode, autorecon2_vol, [('subjects_dir', 'subjects_dir'),
                                     ('subject_id', 'subject_id')]),
        (autorecon2_vol, autorecon_surfs, [('subjects_dir', 'subjects_dir'),
                                           ('subject_id', 'subject_id')]),
        (autorecon_surfs, autorecon3, [(('subjects_dir', _dedup), 'subjects_dir'),
                                       (('subject_id', _dedup), 'subject_id')]),
        (autorecon3, outputnode, [(('subjects_dir', _dedup), 'subjects_dir'),
                                  (('subject_id', _dedup), 'subject_id')]),
        (autorecon3, recon_report, [(('subjects_dir', _dedup), 'subjects_dir'),
                                    (('subject_id', _dedup), 'subject_id')]),
        (recon_report, outputnode, [('out_report', 'out_report')]),
    ])

    return workflow


def init_gifti_surface_wf(name='gifti_surface_wf'):
    r"""
    This workflow prepares GIFTI surfaces from a FreeSurfer subjects directory

    If midthickness (or graymid) surfaces do not exist, they are generated and
    saved to the subject directory as ``lh/rh.midthickness``.
    These, along with the gray/white matter boundary (``lh/rh.smoothwm``), pial
    sufaces (``lh/rh.pial``) and inflated surfaces (``lh/rh.inflated``) are
    converted to GIFTI files.
    Additionally, the vertex coordinates are :py:class:`recentered
    <fmriprep.interfaces.NormalizeSurf>` to align with native T1w space.

    .. workflow::
        :graph2use: orig
        :simple_form: yes

        from fmriprep.workflows.anatomical import init_gifti_surface_wf
        wf = init_gifti_surface_wf()

    **Inputs**

        subjects_dir
            FreeSurfer SUBJECTS_DIR
        subject_id
            FreeSurfer subject ID
        t1_2_fsnative_reverse_transform
            LTA formatted affine transform file (inverse)

    **Outputs**

        surfaces
            GIFTI surfaces for gray/white matter boundary, pial surface,
            midthickness (or graymid) surface, and inflated surfaces

    """
    workflow = Workflow(name=name)

    inputnode = pe.Node(niu.IdentityInterface(['subjects_dir', 'subject_id',
                                               't1_2_fsnative_reverse_transform']),
                        name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(['surfaces']), name='outputnode')

    get_surfaces = pe.Node(nio.FreeSurferSource(), name='get_surfaces')

    midthickness = pe.MapNode(
        MakeMidthickness(thickness=True, distance=0.5, out_name='midthickness'),
        iterfield='in_file',
        name='midthickness')

    save_midthickness = pe.Node(nio.DataSink(parameterization=False),
                                name='save_midthickness')

    surface_list = pe.Node(niu.Merge(4, ravel_inputs=True),
                           name='surface_list', run_without_submitting=True)
    fs_2_gii = pe.MapNode(fs.MRIsConvert(out_datatype='gii'),
                          iterfield='in_file', name='fs_2_gii')
    fix_surfs = pe.MapNode(NormalizeSurf(), iterfield='in_file', name='fix_surfs')

    workflow.connect([
        (inputnode, get_surfaces, [('subjects_dir', 'subjects_dir'),
                                   ('subject_id', 'subject_id')]),
        (inputnode, save_midthickness, [('subjects_dir', 'base_directory'),
                                        ('subject_id', 'container')]),
        # Generate midthickness surfaces and save to FreeSurfer derivatives
        (get_surfaces, midthickness, [('smoothwm', 'in_file'),
                                      ('graymid', 'graymid')]),
        (midthickness, save_midthickness, [('out_file', 'surf.@graymid')]),
        # Produce valid GIFTI surface files (dense mesh)
        (get_surfaces, surface_list, [('smoothwm', 'in1'),
                                      ('pial', 'in2'),
                                      ('inflated', 'in3')]),
        (save_midthickness, surface_list, [('out_file', 'in4')]),
        (surface_list, fs_2_gii, [('out', 'in_file')]),
        (fs_2_gii, fix_surfs, [('converted', 'in_file')]),
        (inputnode, fix_surfs, [('t1_2_fsnative_reverse_transform', 'transform_file')]),
        (fix_surfs, outputnode, [('out_file', 'surfaces')]),
    ])
    return workflow


def init_segs_to_native_wf(name='segs_to_native', segmentation='aseg'):
    """
    Get a segmentation from FreeSurfer conformed space into native T1w space


    .. workflow::
        :graph2use: orig
        :simple_form: yes

        from fmriprep.workflows.anatomical import init_segs_to_native_wf
        wf = init_segs_to_native_wf()


    **Parameters**
        segmentation
            The name of a segmentation ('aseg' or 'aparc_aseg' or 'wmparc')

    **Inputs**

        in_file
            Anatomical, merged T1w image after INU correction
        subjects_dir
            FreeSurfer SUBJECTS_DIR
        subject_id
            FreeSurfer subject ID


    **Outputs**

        out_file
            The selected segmentation, after resampling in native space
    """
    workflow = Workflow(name='%s_%s' % (name, segmentation))
    inputnode = pe.Node(niu.IdentityInterface([
        'in_file', 'subjects_dir', 'subject_id']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(['out_file']), name='outputnode')
    # Extract the aseg and aparc+aseg outputs
    fssource = pe.Node(nio.FreeSurferSource(), name='fs_datasource')
    tonative = pe.Node(fs.Label2Vol(), name='tonative')
    tonii = pe.Node(fs.MRIConvert(out_type='niigz', resample_type='nearest'), name='tonii')

    if segmentation.startswith('aparc'):
        if segmentation == 'aparc_aseg':
            def _sel(x): return x[0]
        elif segmentation == 'aparc_a2009s':
            def _sel(x): return x[1]
        elif segmentation == 'aparc_dkt':
            def _sel(x): return x[2]
        segmentation = (segmentation, _sel)

    workflow.connect([
        (inputnode, fssource, [
            ('subjects_dir', 'subjects_dir'),
            ('subject_id', 'subject_id')]),
        (inputnode, tonii, [('in_file', 'reslice_like')]),
        (fssource, tonative, [(segmentation, 'seg_file'),
                              ('rawavg', 'template_file'),
                              ('aseg', 'reg_header')]),
        (tonative, tonii, [('vol_label_file', 'in_file')]),
        (tonii, outputnode, [('out_file', 'out_file')]),
    ])
    return workflow


def init_anat_reports_wf(reportlets_dir, output_spaces,
                         template, freesurfer, name='anat_reports_wf'):
    """
    Set up a battery of datasinks to store reports in the right location
    """
    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=['source_file', 't1_conform_report', 'seg_report',
                    't1_2_mni_report', 'recon_report']),
        name='inputnode')

    ds_t1_conform_report = pe.Node(
        DerivativesDataSink(base_directory=reportlets_dir, suffix='conform'),
        name='ds_t1_conform_report', run_without_submitting=True)

    ds_t1_2_mni_report = pe.Node(
        DerivativesDataSink(base_directory=reportlets_dir, suffix='t1_2_mni'),
        name='ds_t1_2_mni_report', run_without_submitting=True)

    ds_t1_seg_mask_report = pe.Node(
        DerivativesDataSink(base_directory=reportlets_dir, suffix='seg_brainmask'),
        name='ds_t1_seg_mask_report', run_without_submitting=True)

    ds_recon_report = pe.Node(
        DerivativesDataSink(base_directory=reportlets_dir, suffix='reconall'),
        name='ds_recon_report', run_without_submitting=True)

    workflow.connect([
        (inputnode, ds_t1_conform_report, [('source_file', 'source_file'),
                                           ('t1_conform_report', 'in_file')]),
        (inputnode, ds_t1_seg_mask_report, [('source_file', 'source_file'),
                                            ('seg_report', 'in_file')]),
    ])

    if freesurfer:
        workflow.connect([
            (inputnode, ds_recon_report, [('source_file', 'source_file'),
                                          ('recon_report', 'in_file')])
        ])
    if 'template' in output_spaces:
        workflow.connect([
            (inputnode, ds_t1_2_mni_report, [('source_file', 'source_file'),
                                             ('t1_2_mni_report', 'in_file')])
        ])

    return workflow


def init_anat_derivatives_wf(output_dir, output_spaces, template, freesurfer,
                             name='anat_derivatives_wf'):
    """
    Set up a battery of datasinks to store derivatives in the right location
    """
    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=['source_files', 't1_template_transforms',
                    't1_preproc', 't1_mask', 't1_seg', 't1_tpms',
                    't1_2_mni_forward_transform', 't1_2_mni_reverse_transform',
                    ### SM adjusted: add coordinate mappings!
                    't1_2_mni_forward_transform_cmap', 't1_2_mni_reverse_transform_cmap',
                    't1_2_mni', 'mni_mask', 'mni_seg', 'mni_tpms',
                    't1_2_fsnative_forward_transform', 'surfaces',
                    't1_fs_aseg', 't1_fs_aparc']),
        name='inputnode')

    t1_name = pe.Node(niu.Function(function=fix_multi_T1w_source_name), name='t1_name')

    ds_t1_preproc = pe.Node(
        DerivativesDataSink(base_directory=output_dir, suffix='preproc'),
        name='ds_t1_preproc', run_without_submitting=True)

    ds_t1_mask = pe.Node(
        DerivativesDataSink(base_directory=output_dir, suffix='brainmask'),
        name='ds_t1_mask', run_without_submitting=True)

    ds_t1_seg = pe.Node(
        DerivativesDataSink(base_directory=output_dir, suffix='dtissue'),
        name='ds_t1_seg', run_without_submitting=True)

    ds_t1_tpms = pe.Node(
        DerivativesDataSink(base_directory=output_dir,
                            suffix='class-{extra_value}_probtissue'),
        name='ds_t1_tpms', run_without_submitting=True)
    ds_t1_tpms.inputs.extra_values = ['CSF', 'GM', 'WM']

    suffix_fmt = 'space-{}_{}'.format
    ds_t1_mni = pe.Node(
        DerivativesDataSink(base_directory=output_dir,
                            suffix=suffix_fmt(template, 'preproc')),
        name='ds_t1_mni', run_without_submitting=True)

    ds_mni_mask = pe.Node(
        DerivativesDataSink(base_directory=output_dir,
                            suffix=suffix_fmt(template, 'brainmask')),
        name='ds_mni_mask', run_without_submitting=True)

    ds_mni_seg = pe.Node(
        DerivativesDataSink(base_directory=output_dir,
                            suffix=suffix_fmt(template, 'dtissue')),
        name='ds_mni_seg', run_without_submitting=True)

    ds_mni_tpms = pe.Node(
        DerivativesDataSink(base_directory=output_dir,
                            suffix=suffix_fmt(template, 'class-{extra_value}_probtissue')),
        name='ds_mni_tpms', run_without_submitting=True)
    ds_mni_tpms.inputs.extra_values = ['CSF', 'GM', 'WM']

    # Transforms
    suffix_fmt = 'space-{}_target-{}_{}'.format
    ds_t1_mni_inv_warp = pe.Node(
        DerivativesDataSink(base_directory=output_dir,
                            suffix=suffix_fmt(template, 'T1w', 'warp')),
        name='ds_t1_mni_inv_warp', run_without_submitting=True)

    ###### SM save transform as coordinate mapping as well
    ds_t1_mni_inv_warp_cm = pe.Node(
        DerivativesDataSink(base_directory=output_dir,
                            suffix=suffix_fmt(template, 'T1w', 'warp-coordmapping')),
        name='ds_t1_mni_inv_warp_coordmapping', run_without_submitting=True)

    ds_t1_template_transforms = pe.MapNode(
        DerivativesDataSink(base_directory=output_dir, suffix=suffix_fmt('orig', 'T1w', 'affine')),
        iterfield=['source_file', 'in_file'],
        name='ds_t1_template_transforms', run_without_submitting=True)

    suffix_fmt = 'target-{}_{}'.format
    ds_t1_mni_warp = pe.Node(
        DerivativesDataSink(base_directory=output_dir, suffix=suffix_fmt(template, 'warp')),
        name='ds_t1_mni_warp', run_without_submitting=True)

    ####### SM save transform as coordinate mapping as well
    ds_t1_mni_warp_cm = pe.Node(
        DerivativesDataSink(base_directory=output_dir,
                            suffix=suffix_fmt(template, 'warp-coordmapping')),
        name='ds_t1_mni_warp_coordmapping', run_without_submitting=True)


    lta_2_itk = pe.Node(fs.utils.LTAConvert(out_itk=True), name='lta_2_itk')

    ds_t1_fsnative = pe.Node(
        DerivativesDataSink(base_directory=output_dir, suffix=suffix_fmt('fsnative', 'affine')),
        name='ds_t1_fsnative', run_without_submitting=True)

    name_surfs = pe.MapNode(GiftiNameSource(pattern=r'(?P<LR>[lr])h.(?P<surf>.+)_converted.gii',
                                            template='{surf}.{LR}.surf'),
                            iterfield='in_file',
                            name='name_surfs',
                            run_without_submitting=True)

    ds_surfs = pe.MapNode(
        DerivativesDataSink(base_directory=output_dir),
        iterfield=['in_file', 'suffix'], name='ds_surfs', run_without_submitting=True)

    workflow.connect([
        (inputnode, t1_name, [('source_files', 'in_files')]),
        (inputnode, ds_t1_template_transforms, [('source_files', 'source_file'),
                                                ('t1_template_transforms', 'in_file')]),
        (inputnode, ds_t1_preproc, [('t1_preproc', 'in_file')]),
        (inputnode, ds_t1_mask, [('t1_mask', 'in_file')]),
        (inputnode, ds_t1_seg, [('t1_seg', 'in_file')]),
        (inputnode, ds_t1_tpms, [('t1_tpms', 'in_file')]),
        (t1_name, ds_t1_preproc, [('out', 'source_file')]),
        (t1_name, ds_t1_mask, [('out', 'source_file')]),
        (t1_name, ds_t1_seg, [('out', 'source_file')]),
        (t1_name, ds_t1_tpms, [('out', 'source_file')]),
    ])

    if freesurfer:
        ds_t1_fsaseg = pe.Node(
            DerivativesDataSink(base_directory=output_dir, suffix='label-aseg_roi'),
            name='ds_t1_fsaseg', run_without_submitting=True)
        ds_t1_fsparc = pe.Node(
            DerivativesDataSink(base_directory=output_dir, suffix='label-aparcaseg_roi'),
            name='ds_t1_fsparc', run_without_submitting=True)
        workflow.connect([
            (inputnode, lta_2_itk, [('t1_2_fsnative_forward_transform', 'in_lta')]),
            (t1_name, ds_t1_fsnative, [('out', 'source_file')]),
            (lta_2_itk, ds_t1_fsnative, [('out_itk', 'in_file')]),
            (inputnode, name_surfs, [('surfaces', 'in_file')]),
            (inputnode, ds_surfs, [('surfaces', 'in_file')]),
            (t1_name, ds_surfs, [('out', 'source_file')]),
            (name_surfs, ds_surfs, [('out_name', 'suffix')]),
            (inputnode, ds_t1_fsaseg, [('t1_fs_aseg', 'in_file')]),
            (inputnode, ds_t1_fsparc, [('t1_fs_aparc', 'in_file')]),
            (t1_name, ds_t1_fsaseg, [('out', 'source_file')]),
            (t1_name, ds_t1_fsparc, [('out', 'source_file')]),
        ])
    if 'template' in output_spaces:
        workflow.connect([
            (inputnode, ds_t1_mni_warp, [('t1_2_mni_forward_transform', 'in_file')]),
            (inputnode, ds_t1_mni_warp_cm, [('t1_2_mni_forward_transform_cmap', 'in_file')]),   ## Added cmap
            (inputnode, ds_t1_mni_inv_warp, [('t1_2_mni_reverse_transform', 'in_file')]),
            (inputnode, ds_t1_mni_inv_warp_cm, [('t1_2_mni_reverse_transform_cmap', 'in_file')]),   ## Added cmap
            (inputnode, ds_t1_mni, [('t1_2_mni', 'in_file')]),
            (inputnode, ds_mni_mask, [('mni_mask', 'in_file')]),
            (inputnode, ds_mni_seg, [('mni_seg', 'in_file')]),
            (inputnode, ds_mni_tpms, [('mni_tpms', 'in_file')]),
            (t1_name, ds_t1_mni_warp, [('out', 'source_file')]),
            (t1_name, ds_t1_mni_inv_warp, [('out', 'source_file')]),
            (t1_name, ds_t1_mni, [('out', 'source_file')]),
            (t1_name, ds_mni_mask, [('out', 'source_file')]),
            (t1_name, ds_mni_seg, [('out', 'source_file')]),
            (t1_name, ds_mni_tpms, [('out', 'source_file')]),
        ])

    return workflow


def _seg2msks(in_file, newpath=None):
    """Converts labels to masks"""
    import nibabel as nb
    import numpy as np
    from nipype.utils.filemanip import fname_presuffix

    nii = nb.load(in_file)
    labels = nii.get_data()

    out_files = []
    for i in range(1, 4):
        ldata = np.zeros_like(labels)
        ldata[labels == i] = 1
        out_files.append(fname_presuffix(
            in_file, suffix='_label%03d' % i, newpath=newpath))
        nii.__class__(ldata, nii.affine, nii.header).to_filename(out_files[-1])

    return out_files


### SM: the following functions allow for skull stripping & registration
def get_first(x):
    if isinstance(x, list):
        return x[0]
    else:
        return x


def skull_strip(second_inversion, t1_weighted, t1_map):
    """ Wrapper function for nighres.brain.mp2rage_skullstripping for compatibility with nipype """
    import nighres
    import os
    import nibabel as nib
    file_name = os.path.basename(t1_weighted).split('_T1w')[0]

    res = nighres.brain.mp2rage_skullstripping(second_inversion, t1_weighted, t1_map,
                                               save_data=True)  # save manually

    fn_dict = {}
    for key, val in res.items():
        nib.save(val, os.path.abspath('./' + file_name + '_' + key.replace('_', '-') + '.nii.gz'))
        fn_dict[key] = os.path.abspath('./' + file_name + '_' + key.replace('_', '-') + '.nii.gz')
    # print(fn_dict)

    return fn_dict['brain_mask'], fn_dict['inv2_masked'], fn_dict['t1w_masked'], fn_dict['t1map_masked']


def register_func(source_img, target_img, run_rigid=True, run_syn=True, run_affine=True, compress_report=False, *args):
    """ Wrapper function for nighres.registration.embedded_antsreg for compatibility with nipype.
    *args is allowed as input so that any arguments usually used in the fmriprep-anatomical pipeline don't cause
    errors. They are not used.
    """
    ########### SM: customized version of embedded_antsreg below. Ugly as hell, I know. Sorry.
    # basic dependencies
    import os
    import sys
    import subprocess
    from glob import glob

    # main dependencies: numpy, nibabel
    import numpy as np
    import nibabel as nb

    # nighresjava and nighres functions
    import nighresjava
    from nighres.io import load_volume, save_volume
    from nighres.utils import _output_dir_4saving, _fname_4saving, \
        _check_topology_lut_dir

    # convenience labels
    X = 0
    Y = 1
    Z = 2
    T = 3

    def embedded_antsregSM(source_image, target_image,
                           run_rigid=True,
                           rigid_iterations=1000,
                           run_affine=False,
                           affine_iterations=1000,
                           run_syn=True,
                           coarse_iterations=40,
                           medium_iterations=50, fine_iterations=40,
                           cost_function='MutualInformation',
                           interpolation='NearestNeighbor',
                           regularization='High',
                           convergence=1e-6,
                           ignore_affine=False, ignore_header=False,
                           save_data=False, overwrite=False, output_dir=None,
                           file_name=None):
        """ Embedded ANTS Registration
        Runs the rigid and/or Symmetric Normalization (SyN) algorithm of ANTs and
        formats the output deformations into voxel coordinate mappings as used in
        CBSTools registration and transformation routines.
        Parameters
        ----------
        source_image: niimg
            Image to register
        target_image: niimg
            Reference image to match
        run_rigid: bool
            Whether or not to run a rigid registration first (default is False)
        rigid_iterations: float
            Number of iterations in the rigid step (default is 1000)
        run_affine: bool
            Whether or not to run a affine registration first (default is False)
        affine_iterations: float
            Number of iterations in the affine step (default is 1000)
        run_syn: bool
            Whether or not to run a SyN registration (default is True)
        coarse_iterations: float
            Number of iterations at the coarse level (default is 40)
        medium_iterations: float
            Number of iterations at the medium level (default is 50)
        fine_iterations: float
            Number of iterations at the fine level (default is 40)
        cost_function: {'CrossCorrelation', 'MutualInformation'}
            Cost function for the registration (default is 'MutualInformation')
        interpolation: {'NearestNeighbor', 'Linear'}
            Interpolation for the registration result (default is 'NearestNeighbor')
        regularization: {'Low', 'Medium', 'High'}
            Regularization preset for the SyN deformation (default is 'Medium')
        convergence: float
            Threshold for convergence, can make the algorithm very slow (default is convergence)
        ignore_affine: bool
            Ignore the affine matrix information extracted from the image header
            (default is False)
        ignore_header: bool
            Ignore the orientation information and affine matrix information
            extracted from the image header (default is False)
        save_data: bool
            Save output data to file (default is False)
        overwrite: bool
            Overwrite existing results (default is False)
        output_dir: str, optional
            Path to desired output directory, will be created if it doesn't exist
        file_name: str, optional
            Desired base name for output files with file extension
            (suffixes will be added)
        Returns
        ----------
        dict
            Dictionary collecting outputs under the following keys
            (suffix of output files in brackets)
            * transformed_source (niimg): Deformed source image (_ants_def)
            * mapping (niimg): Coordinate mapping from source to target (_ants_map)
            * inverse (niimg): Inverse coordinate mapping from target to source (_ants_invmap)
        Notes
        ----------
        Port of the CBSTools Java module by Pierre-Louis Bazin. The main algorithm
        is part of the ANTs software by Brian Avants and colleagues [1]_. The interfacing
        with ANTs is performed through Nipype [2]_. Parameters have been set to values
        commonly found in neuroimaging scripts online, but not necessarily optimal.
        References
        ----------
        .. [1] Avants et al (2008), Symmetric diffeomorphic
           image registration with cross-correlation: evaluating automated labeling
           of elderly and neurodegenerative brain, Med Image Anal. 12(1):26-41
        .. [2] Gorgolewski et al (2011) Nipype: a flexible, lightweight and extensible
           neuroimaging data processing framework in python. Front Neuroinform 5.
           doi:10.3389/fninf.2011.00013
        """

        print('\nEmbedded ANTs Registration')

        # for external tools: nipype
        try:
            from nipype.interfaces.ants import Registration
            from nipype.interfaces.ants import ApplyTransforms
        except ImportError:
            print('Error: Nipype and/or ANTS could not be imported, they are required'
                  + 'in order to run this module. \n (aborting)')
            return None

        # make sure that saving related parameters are correct
        output_dir = _output_dir_4saving(output_dir, source_image)  # needed for intermediate results
        if save_data:
            transformed_source_file = os.path.join(output_dir,
                                                   _fname_4saving(file_name=file_name,
                                                                  rootfile=source_image,
                                                                  suffix='ants-def'))

            mapping_file = os.path.join(output_dir,
                                        _fname_4saving(file_name=file_name,
                                                       rootfile=source_image,
                                                       suffix='ants-map'))

            inverse_mapping_file = os.path.join(output_dir,
                                                _fname_4saving(file_name=file_name,
                                                               rootfile=source_image,
                                                               suffix='ants-invmap'))
            if overwrite is False \
                    and os.path.isfile(transformed_source_file) \
                    and os.path.isfile(mapping_file) \
                    and os.path.isfile(inverse_mapping_file):
                print("skip computation (use existing results)")
                output = {'transformed_source': load_volume(transformed_source_file),
                          'mapping': load_volume(mapping_file),
                          'inverse': load_volume(inverse_mapping_file)}
                return output

        # load and get dimensions and resolution from input images
        source = load_volume(source_image)
        src_affine = source.affine
        src_header = source.header
        nsx = source.header.get_data_shape()[X]
        nsy = source.header.get_data_shape()[Y]
        nsz = source.header.get_data_shape()[Z]
        rsx = source.header.get_zooms()[X]
        rsy = source.header.get_zooms()[Y]
        rsz = source.header.get_zooms()[Z]

        target = load_volume(target_image)
        trg_affine = target.affine
        trg_header = target.header
        ntx = target.header.get_data_shape()[X]
        nty = target.header.get_data_shape()[Y]
        ntz = target.header.get_data_shape()[Z]
        rtx = target.header.get_zooms()[X]
        rty = target.header.get_zooms()[Y]
        rtz = target.header.get_zooms()[Z]

        # in case the affine transformations are not to be trusted: make them equal
        if ignore_affine or ignore_header:
            # create generic affine aligned with the orientation for the source
            mx = np.argmax(np.abs(src_affine[0][0:3]))
            my = np.argmax(np.abs(src_affine[1][0:3]))
            mz = np.argmax(np.abs(src_affine[2][0:3]))
            new_affine = np.zeros((4, 4))
            if ignore_header:
                new_affine[0][0] = rsx
                new_affine[1][1] = rsy
                new_affine[2][2] = rsz
                new_affine[0][3] = -rsx * nsx / 2.0
                new_affine[1][3] = -rsy * nsy / 2.0
                new_affine[2][3] = -rsz * nsz / 2.0
            else:
                new_affine[0][mx] = rsx * np.sign(src_affine[0][mx])
                new_affine[1][my] = rsy * np.sign(src_affine[1][my])
                new_affine[2][mz] = rsz * np.sign(src_affine[2][mz])
                if (np.sign(src_affine[0][mx]) < 0):
                    new_affine[0][3] = rsx * nsx / 2.0
                else:
                    new_affine[0][3] = -rsx * nsx / 2.0

                if (np.sign(src_affine[1][my]) < 0):
                    new_affine[1][3] = rsy * nsy / 2.0
                else:
                    new_affine[1][3] = -rsy * nsy / 2.0

                if (np.sign(src_affine[2][mz]) < 0):
                    new_affine[2][3] = rsz * nsz / 2.0
                else:
                    new_affine[2][3] = -rsz * nsz / 2.0
            # if (np.sign(src_affine[0][mx])<0): new_affine[mx][3] = rsx*nsx
            # if (np.sign(src_affine[1][my])<0): new_affine[my][3] = rsy*nsy
            # if (np.sign(src_affine[2][mz])<0): new_affine[mz][3] = rsz*nsz
            # new_affine[0][3] = nsx/2.0
            # new_affine[1][3] = nsy/2.0
            # new_affine[2][3] = nsz/2.0
            new_affine[3][3] = 1.0

            src_img = nb.Nifti1Image(source.get_data(), new_affine, source.header)
            src_img.update_header()
            src_img_file = os.path.join(output_dir, _fname_4saving(file_name=file_name,
                                                                   rootfile=source_image,
                                                                   suffix='tmp_srcimg'))
            save_volume(src_img_file, src_img)
            source = load_volume(src_img_file)
            src_affine = source.affine
            src_header = source.header

            # create generic affine aligned with the orientation for the target
            mx = np.argmax(np.abs(trg_affine[0][0:3]))
            my = np.argmax(np.abs(trg_affine[1][0:3]))
            mz = np.argmax(np.abs(trg_affine[2][0:3]))
            new_affine = np.zeros((4, 4))
            if ignore_header:
                new_affine[0][0] = rtx
                new_affine[1][1] = rty
                new_affine[2][2] = rtz
                new_affine[0][3] = -rtx * ntx / 2.0
                new_affine[1][3] = -rty * nty / 2.0
                new_affine[2][3] = -rtz * ntz / 2.0
            else:
                new_affine[0][mx] = rtx * np.sign(trg_affine[0][mx])
                new_affine[1][my] = rty * np.sign(trg_affine[1][my])
                new_affine[2][mz] = rtz * np.sign(trg_affine[2][mz])
                if (np.sign(trg_affine[0][mx]) < 0):
                    new_affine[0][3] = rtx * ntx / 2.0
                else:
                    new_affine[0][3] = -rtx * ntx / 2.0

                if (np.sign(trg_affine[1][my]) < 0):
                    new_affine[1][3] = rty * nty / 2.0
                else:
                    new_affine[1][3] = -rty * nty / 2.0

                if (np.sign(trg_affine[2][mz]) < 0):
                    new_affine[2][3] = rtz * ntz / 2.0
                else:
                    new_affine[2][3] = -rtz * ntz / 2.0
            # if (np.sign(trg_affine[0][mx])<0): new_affine[mx][3] = rtx*ntx
            # if (np.sign(trg_affine[1][my])<0): new_affine[my][3] = rty*nty
            # if (np.sign(trg_affine[2][mz])<0): new_affine[mz][3] = rtz*ntz
            # new_affine[0][3] = ntx/2.0
            # new_affine[1][3] = nty/2.0
            # new_affine[2][3] = ntz/2.0
            new_affine[3][3] = 1.0

            trg_img = nb.Nifti1Image(target.get_data(), new_affine, target.header)
            trg_img.update_header()
            trg_img_file = os.path.join(output_dir, _fname_4saving(file_name=file_name,
                                                                   rootfile=source_image,
                                                                   suffix='tmp_trgimg'))
            save_volume(trg_img_file, trg_img)
            target = load_volume(trg_img_file)
            trg_affine = target.affine
            trg_header = target.header

        # build coordinate mapping matrices and save them to disk
        src_coord = np.zeros((nsx, nsy, nsz, 3))
        trg_coord = np.zeros((ntx, nty, ntz, 3))
        for x in range(nsx):
            for y in range(nsy):
                for z in range(nsz):
                    src_coord[x, y, z, X] = x
                    src_coord[x, y, z, Y] = y
                    src_coord[x, y, z, Z] = z
        src_map = nb.Nifti1Image(src_coord, source.affine, source.header)
        src_map_file = os.path.join(output_dir, _fname_4saving(file_name=file_name,
                                                               rootfile=source_image,
                                                               suffix='tmp_srccoord'))
        save_volume(src_map_file, src_map)
        for x in range(ntx):
            for y in range(nty):
                for z in range(ntz):
                    trg_coord[x, y, z, X] = x
                    trg_coord[x, y, z, Y] = y
                    trg_coord[x, y, z, Z] = z
        trg_map = nb.Nifti1Image(trg_coord, target.affine, target.header)
        trg_map_file = os.path.join(output_dir, _fname_4saving(file_name=file_name,
                                                               rootfile=source_image,
                                                               suffix='tmp_trgcoord'))
        save_volume(trg_map_file, trg_map)

        # run the main ANTS software
        reg = Registration()
        reg.inputs.dimension = 3

        # add a prefix to avoid multiple names?
        prefix = _fname_4saving(file_name=file_name,
                                rootfile=source_image,
                                suffix='tmp_syn')
        prefix = os.path.basename(prefix)
        prefix = prefix.split(".")[0]
        reg.inputs.output_transform_prefix = prefix
        reg.inputs.fixed_image = [target.get_filename()]
        reg.inputs.moving_image = [source.get_filename()]

        print("registering " + source.get_filename() + "\n to " + target.get_filename())

        if run_syn is True:
            if regularization is 'Low':
                syn_param = (0.2, 1.0, 0.0)
            elif regularization is 'Medium':
                syn_param = (0.2, 3.0, 0.0)
            elif regularization is 'High':
                syn_param = (0.2, 4.0, 3.0)
            else:
                syn_param = (0.2, 3.0, 0.0)

        if run_rigid is True and run_affine is True and run_syn is True:
            reg.inputs.transforms = ['Rigid', 'Affine', 'SyN']
            reg.inputs.transform_parameters = [(0.1,), (0.1,), syn_param]
            reg.inputs.number_of_iterations = [[rigid_iterations, rigid_iterations,
                                                rigid_iterations],
                                               [affine_iterations, affine_iterations,
                                                affine_iterations],
                                               [coarse_iterations, coarse_iterations,
                                                medium_iterations, fine_iterations]]
            if (cost_function == 'CrossCorrelation'):
                reg.inputs.metric = ['CC', 'CC', 'CC']
                reg.inputs.metric_weight = [1.0, 1.0, 1.0]
                reg.inputs.radius_or_number_of_bins = [5, 5, 5]
            else:
                reg.inputs.metric = ['MI', 'MI', 'MI']
                reg.inputs.metric_weight = [1.0, 1.0, 1.0]
                reg.inputs.radius_or_number_of_bins = [32, 32, 32]
            reg.inputs.shrink_factors = [[4, 2, 1]] + [[4, 2, 1]] + [[8, 4, 2, 1]]
            reg.inputs.smoothing_sigmas = [[3, 2, 1]] + [[3, 2, 1]] + [[2, 1, 0.5, 0]]
            reg.inputs.sampling_strategy = ['Random'] + ['Random'] + ['Random']
            reg.inputs.sampling_percentage = [0.3] + [0.3] + [0.3]
            reg.inputs.convergence_threshold = [convergence] + [convergence] + [convergence]
            reg.inputs.convergence_window_size = [10] + [10] + [5]
            reg.inputs.use_histogram_matching = [False] + [False] + [False]
            reg.inputs.winsorize_lower_quantile = 0.001
            reg.inputs.winsorize_upper_quantile = 0.999

        elif run_rigid is True and run_affine is False and run_syn is True:
            reg.inputs.transforms = ['Rigid', 'SyN']
            reg.inputs.transform_parameters = [(0.1,), syn_param]
            reg.inputs.number_of_iterations = [[rigid_iterations, rigid_iterations,
                                                rigid_iterations],
                                               [coarse_iterations, coarse_iterations,
                                                medium_iterations, fine_iterations]]
            if (cost_function == 'CrossCorrelation'):
                reg.inputs.metric = ['CC', 'CC']
                reg.inputs.metric_weight = [1.0, 1.0]
                reg.inputs.radius_or_number_of_bins = [5, 5]
            else:
                reg.inputs.metric = ['MI', 'MI']
                reg.inputs.metric_weight = [1.0, 1.0]
                reg.inputs.radius_or_number_of_bins = [32, 32]
            reg.inputs.shrink_factors = [[4, 2, 1]] + [[8, 4, 2, 1]]
            reg.inputs.smoothing_sigmas = [[3, 2, 1]] + [[2, 1, 0.5, 0]]
            reg.inputs.sampling_strategy = ['Random'] + ['Random']
            reg.inputs.sampling_percentage = [0.3] + [0.3]
            reg.inputs.convergence_threshold = [convergence] + [convergence]
            reg.inputs.convergence_window_size = [10] + [5]
            reg.inputs.use_histogram_matching = [False] + [False]
            reg.inputs.winsorize_lower_quantile = 0.001
            reg.inputs.winsorize_upper_quantile = 0.999

        elif run_rigid is False and run_affine is True and run_syn is True:
            reg.inputs.transforms = ['Affine', 'SyN']
            reg.inputs.transform_parameters = [(0.1,), syn_param]
            reg.inputs.number_of_iterations = [[affine_iterations, affine_iterations,
                                                affine_iterations],
                                               [coarse_iterations, coarse_iterations,
                                                medium_iterations, fine_iterations]]
            if (cost_function == 'CrossCorrelation'):
                reg.inputs.metric = ['CC', 'CC']
                reg.inputs.metric_weight = [1.0, 1.0]
                reg.inputs.radius_or_number_of_bins = [5, 5]
            else:
                reg.inputs.metric = ['MI', 'MI']
                reg.inputs.metric_weight = [1.0, 1.0]
                reg.inputs.radius_or_number_of_bins = [64, 64]
            reg.inputs.shrink_factors = [[4, 2, 1]] + [[8, 4, 2, 1]]
            reg.inputs.smoothing_sigmas = [[3, 2, 1]] + [[2, 1, 0.5, 0]]
            reg.inputs.sampling_strategy = ['Random'] + ['Random']
            reg.inputs.sampling_percentage = [0.3] + [0.3]
            reg.inputs.convergence_threshold = [convergence] + [convergence]
            reg.inputs.convergence_window_size = [10] + [5]
            reg.inputs.use_histogram_matching = [False] + [False]
            reg.inputs.winsorize_lower_quantile = 0.001
            reg.inputs.winsorize_upper_quantile = 0.999

        if run_rigid is True and run_affine is True and run_syn is False:
            reg.inputs.transforms = ['Rigid', 'Affine']
            reg.inputs.transform_parameters = [(0.1,), (0.1,)]
            reg.inputs.number_of_iterations = [[rigid_iterations, rigid_iterations,
                                                rigid_iterations],
                                               [affine_iterations, affine_iterations,
                                                affine_iterations]]
            if (cost_function == 'CrossCorrelation'):
                reg.inputs.metric = ['CC', 'CC']
                reg.inputs.metric_weight = [1.0, 1.0]
                reg.inputs.radius_or_number_of_bins = [5, 5]
            else:
                reg.inputs.metric = ['MI', 'MI']
                reg.inputs.metric_weight = [1.0, 1.0]
                reg.inputs.radius_or_number_of_bins = [32, 32]
            reg.inputs.shrink_factors = [[4, 2, 1]] + [[4, 2, 1]]
            reg.inputs.smoothing_sigmas = [[3, 2, 1]] + [[3, 2, 1]]
            reg.inputs.sampling_strategy = ['Random'] + ['Random']
            reg.inputs.sampling_percentage = [0.3] + [0.3]
            reg.inputs.convergence_threshold = [convergence] + [convergence]
            reg.inputs.convergence_window_size = [10] + [10]
            reg.inputs.use_histogram_matching = [False] + [False]
            reg.inputs.winsorize_lower_quantile = 0.001
            reg.inputs.winsorize_upper_quantile = 0.999

        elif run_rigid is True and run_affine is False and run_syn is False:
            reg.inputs.transforms = ['Rigid']
            reg.inputs.transform_parameters = [(0.1,)]
            reg.inputs.number_of_iterations = [[rigid_iterations, rigid_iterations,
                                                rigid_iterations]]
            if (cost_function == 'CrossCorrelation'):
                reg.inputs.metric = ['CC']
                reg.inputs.metric_weight = [1.0]
                reg.inputs.radius_or_number_of_bins = [5]
            else:
                reg.inputs.metric = ['MI']
                reg.inputs.metric_weight = [1.0]
                reg.inputs.radius_or_number_of_bins = [32]
            reg.inputs.shrink_factors = [[4, 2, 1]]
            reg.inputs.smoothing_sigmas = [[3, 2, 1]]
            reg.inputs.sampling_strategy = ['Random']
            reg.inputs.sampling_percentage = [0.3]
            reg.inputs.convergence_threshold = [convergence]
            reg.inputs.convergence_window_size = [10]
            reg.inputs.use_histogram_matching = [False]
            reg.inputs.winsorize_lower_quantile = 0.001
            reg.inputs.winsorize_upper_quantile = 0.999

        elif run_rigid is False and run_affine is True and run_syn is False:
            reg.inputs.transforms = ['Affine']
            reg.inputs.transform_parameters = [(0.1,)]
            reg.inputs.number_of_iterations = [[affine_iterations, affine_iterations,
                                                affine_iterations]]
            if (cost_function == 'CrossCorrelation'):
                reg.inputs.metric = ['CC']
                reg.inputs.metric_weight = [1.0]
                reg.inputs.radius_or_number_of_bins = [5]
            else:
                reg.inputs.metric = ['MI']
                reg.inputs.metric_weight = [1.0]
                reg.inputs.radius_or_number_of_bins = [32]
            reg.inputs.shrink_factors = [[4, 2, 1]]
            reg.inputs.smoothing_sigmas = [[3, 2, 1]]
            reg.inputs.sampling_strategy = ['Random']
            reg.inputs.sampling_percentage = [0.3]
            reg.inputs.convergence_threshold = [convergence]
            reg.inputs.convergence_window_size = [10]
            reg.inputs.use_histogram_matching = [False]
            reg.inputs.winsorize_lower_quantile = 0.001
            reg.inputs.winsorize_upper_quantile = 0.999

        elif run_rigid is False and run_affine is False and run_syn is True:
            reg.inputs.transforms = ['SyN']
            reg.inputs.transform_parameters = [syn_param]
            reg.inputs.number_of_iterations = [[coarse_iterations, coarse_iterations,
                                                medium_iterations, fine_iterations]]
            if (cost_function == 'CrossCorrelation'):
                reg.inputs.metric = ['CC']
                reg.inputs.metric_weight = [1.0]
                reg.inputs.radius_or_number_of_bins = [5]
            else:
                reg.inputs.metric = ['MI']
                reg.inputs.metric_weight = [1.0]
                reg.inputs.radius_or_number_of_bins = [32]
            reg.inputs.shrink_factors = [[8, 4, 2, 1]]
            reg.inputs.smoothing_sigmas = [[2, 1, 0.5, 0]]
            reg.inputs.sampling_strategy = ['Random']
            reg.inputs.sampling_percentage = [0.3]
            reg.inputs.convergence_threshold = [convergence]
            reg.inputs.convergence_window_size = [10]
            reg.inputs.use_histogram_matching = [False]
            reg.inputs.winsorize_lower_quantile = 0.001
            reg.inputs.winsorize_upper_quantile = 0.999

        elif run_rigid is False and run_affine is False and run_syn is False:
            reg.inputs.transforms = ['Rigid']
            reg.inputs.transform_parameters = [(0.1,)]
            reg.inputs.number_of_iterations = [[0]]
            reg.inputs.metric = ['CC']
            reg.inputs.metric_weight = [1.0]
            reg.inputs.radius_or_number_of_bins = [5]
            reg.inputs.shrink_factors = [[1]]
            reg.inputs.smoothing_sigmas = [[1]]
            reg.inputs.args = '--float 0'

        print(reg.cmdline)
        result = reg.run()

        # Transforms the moving image
        at = ApplyTransforms()
        at.inputs.dimension = 3
        at.inputs.input_image = source.get_filename()
        at.inputs.reference_image = target.get_filename()
        at.inputs.interpolation = interpolation
        at.inputs.transforms = result.outputs.forward_transforms
        at.inputs.invert_transform_flags = result.outputs.forward_invert_flags
        transformed = at.run()

        # Create coordinate mappings
        src_at = ApplyTransforms()
        src_at.inputs.dimension = 3
        src_at.inputs.input_image_type = 3
        src_at.inputs.input_image = src_map.get_filename()
        src_at.inputs.reference_image = target.get_filename()
        src_at.inputs.interpolation = 'Linear'
        src_at.inputs.transforms = result.outputs.forward_transforms
        src_at.inputs.invert_transform_flags = result.outputs.forward_invert_flags
        mapping = src_at.run()

        trg_at = ApplyTransforms()
        trg_at.inputs.dimension = 3
        trg_at.inputs.input_image_type = 3
        trg_at.inputs.input_image = trg_map.get_filename()
        trg_at.inputs.reference_image = source.get_filename()
        trg_at.inputs.interpolation = 'Linear'
        trg_at.inputs.transforms = result.outputs.reverse_transforms
        trg_at.inputs.invert_transform_flags = result.outputs.reverse_invert_flags
        inverse = trg_at.run()

        # pad coordinate mapping outside the image? hopefully not needed...

        # collect outputs and potentially save
        transformed_img = nb.Nifti1Image(nb.load(transformed.outputs.output_image).get_data(),
                                         target.affine, target.header)
        mapping_img = nb.Nifti1Image(nb.load(mapping.outputs.output_image).get_data(),
                                     target.affine, target.header)
        inverse_img = nb.Nifti1Image(nb.load(inverse.outputs.output_image).get_data(),
                                     source.affine, source.header)

        #### SM: also output then ants .mat-files
        outputs = {'transformed_source': transformed_img,
                   'mapping': mapping_img,
                   'inverse': inverse_img,
                   'mapping_ants': result.outputs.forward_transforms,
                   'inverse_ants': result.outputs.reverse_transforms}

        # clean-up intermediate files
        os.remove(src_map_file)
        os.remove(trg_map_file)
        if ignore_affine or ignore_header:
            os.remove(src_img_file)
            os.remove(trg_img_file)

        #### SM: don't remove these tranforms
        # for name in result.outputs.forward_transforms:
        #         if os.path.exists(name): os.remove(name)
        #     for name in result.outputs.reverse_transforms:
        #         if os.path.exists(name): os.remove(name)
        os.remove(transformed.outputs.output_image)
        os.remove(mapping.outputs.output_image)
        os.remove(inverse.outputs.output_image)

        if save_data:
            save_volume(transformed_source_file, transformed_img)
            save_volume(mapping_file, mapping_img)
            save_volume(inverse_mapping_file, inverse_img)

        return outputs


    ## this is where register_func starts ##
    file_name = os.path.basename(source_img)
    if '_echo_' in source_img:
        # generate random string to append to filename, so that multiple echos calling this *exact same function*
        # with *exact same input* do not conflict
        import string
        import random

        def id_generator(size=50, chars=string.ascii_uppercase + string.ascii_lowercase + string.digits):
            return ''.join(random.choice(chars) for _ in range(size))

        file_name = file_name.replace('.nii', '_' + id_generator() + '.nii')
    print(file_name)

    syn_results = embedded_antsregSM(
        source_image=source_img,
        target_image=target_img,
        run_rigid=run_rigid, run_syn=run_syn, run_affine=run_affine,
        file_name=file_name,
        cost_function='MutualInformation',
        interpolation='NearestNeighbor',
        save_data=False, overwrite=False)

    # save coordinate mappings manually
    fn_dict = {}
    for key, val in syn_results.items():
        if isinstance(val, nb.Nifti1Image):
            nb.save(val, os.path.abspath('./' + file_name + '_' + key.replace('_', '-') + '.nii.gz'))
            fn_dict[key] = os.path.abspath('./' + file_name + '_' + key.replace('_', '-') + '.nii.gz')

    inverse_cmap = fn_dict['inverse']
    mapping_cmap = fn_dict['mapping']
    mapping = syn_results['mapping_ants']
    inverse = syn_results['inverse_ants']
    trans_source = fn_dict['transformed_source']

    ## make a nice report for fmriprep ##
    from nipype.interfaces.base import File
    _fixed_image = target_img
    _moving_image = trans_source  # source_img
    _fixed_image_mask = None
    _fixed_image_label = "fixed"
    _moving_image_label = "moving"
    _contour = None
    #    _out_report = File('report.svg', usedefault=True, desc='filename for the visual report')#'report.svg'
    _out_report = os.path.abspath(os.path.join(os.getcwd(), 'report.svg'))

    from niworkflows.viz.utils import plot_registration, compose_view, cuts_from_bbox
    from nilearn.masking import apply_mask, unmask
    from nilearn.image import threshold_img, load_img

    fixed_image_nii = load_img(_fixed_image)  # template
    moving_image_nii = load_img(_moving_image)  # source
    contour_nii = load_img(_contour) if _contour is not None else None

    if _fixed_image_mask:
        fixed_image_nii = unmask(apply_mask(fixed_image_nii,
                                            _fixed_image_mask),
                                 _fixed_image_mask)
        # since the moving image is already in the fixed image space we
        # should apply the same mask
        moving_image_nii = unmask(apply_mask(moving_image_nii,
                                             _fixed_image_mask),
                                  _fixed_image_mask)
        mask_nii = load_img(_fixed_image_mask)
    else:
        mask_nii = threshold_img(fixed_image_nii, 1e-3)

    n_cuts = 7
    if not _fixed_image_mask and contour_nii:
        cuts = cuts_from_bbox(contour_nii, cuts=n_cuts)
    else:
        cuts = cuts_from_bbox(mask_nii, cuts=n_cuts)

    # Call composer
    compose_view(
        plot_registration(fixed_image_nii, 'fixed-image',
                          estimate_brightness=True,
                          cuts=cuts,
                          label=_fixed_image_label,
                          contour=contour_nii,
                          compress=compress_report),
        plot_registration(moving_image_nii, 'moving-image',
                          estimate_brightness=True,
                          cuts=cuts,
                          label=_moving_image_label,
                          contour=contour_nii,
                          compress=compress_report),
        out_file=_out_report
    )
    print(inv)
    print(trans_source)
    print(mapping)
    return trans_source, mapping, inverse, mapping_cmap, inverse_cmap, _out_report



