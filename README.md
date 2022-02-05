# Postprocessing for Diffeomorphic Image Registration
This is a **postprocessing layer** that can be inserted at the end of a deep learning framework that aids in making the registration field more diffeomorphic. In this repository, this particular version of the code takes 3D MRI scans from the OASIS dataset as its input and uses the popular registration framework Voxelmorph. 

# Purpose
Diffeomorphic image registration is important for medical imaging studies because of the properties like invertibility, smoothness of the transformation, and topology preservation/non-folding of the grid. Violation of these properties can lead to destruction of the neighbourhood and the connectivity of anatomical structures during image registration. But recent deep learning models do not explicitly address this **folding problem**. The purpose of this postprocessing layer is to reduce this folding of the registration field.

# Overview
![ICPR_pipeline2](https://user-images.githubusercontent.com/36978751/152610105-0bb01172-8255-407b-8af2-8c5cf1cdc186.png)

For more details, please check [eprint arxiv:2202.00749](https://arxiv.org/abs/2202.00749).

# Reduction of Folding in Action

A demonstration of the effect of our postprocessing layer on the Voxelmorph framework:

![Fig2_ICPR](https://user-images.githubusercontent.com/36978751/152612333-6c2b543c-c73c-435d-8525-3e2e64b48634.svg)

# Paper
If you find our work useful for your research or use some part of the code, please cite :
*   **Towards Positive Jacobian: Learn to Postprocess Diffeomorphic Image Registration with Matrix Exponential.**  
    Soumyadeep Pal, Matthew Tennant, [Nilanjan Ray](https://webdocs.cs.ualberta.ca/~nray1/)    
    [eprint arxiv:2202.00749](https://arxiv.org/abs/2202.00749)


