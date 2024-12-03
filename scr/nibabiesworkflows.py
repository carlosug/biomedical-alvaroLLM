from nibabies.workflows.anatomical import init_anat_preproc_wf

# Initialize NiBabies anatomical preprocessing workflow
nibabies_preproc = init_anat_preproc_wf(
    bids_root="path/to/dataset",  # Input BIDS dataset
    output_dir="path/to/output",
    template="MNIInfant",
    t1w_template_resolution=1.0  # Resolution in mm
)

# Run preprocessing
nibabies_preproc.run()
