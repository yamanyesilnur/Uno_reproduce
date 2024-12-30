# Setup Instructions

Steps to set up the environment and compile the necessary components:

1. **Create the Conda Environment**
    ```bash
    mamba env create -f env.yml
    ```

2. **Compile MSDA**
    ```bash
    cd ./uno_reproduce/Deformable-DETR/models/ops
    bash make.sh
    ```

3. **Install Conda Build and Develop Deformable-DETR**
    ```bash
    conda install conda-build
    conda develop Deformable-DETR
    ```

4. **Run the Mock Forward Pass**
    ```bash
    python mock_forward_pass.py
    ```
    At this point if models cannot be imported error thrown,
    restart the environment, try step 3 again.
