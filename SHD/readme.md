Step 1:
Download the SHD dataset from https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/ and place it in the outermost "data" folder.

Step 2:
run shd_generate_dataset.py to preprocess raw data of SHD first

Step 3:
run main_dense_general_rhy.py for Rhythm-DH-SFNN on the SHD dataset
or run main_dense_orgin.py for vanilla DH-SFNN on the SHD dataset


To start the training of Rhythm-DH-SFNN, you could just run the following commands
  ```
  # Rhythm-DH-SFNN 
  python  main_dense_general_rhy.py
  ```
  or
  ```
  # DH-SFNN
  python  main_dense_orgin.py 
  ``` 
