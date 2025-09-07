# QC LDPC recovery
This project aims to recover H matrix of a LDPC code without candidate set on a noisy channel       

## Installations
This project is based on python. Below are the packages that needs to be installed:

numpy                     
numba                     
scipy                   
matplotlib                   

## Files
- main.py                 
An executable file based on paper 'Progressive reconstruction of large QCLDPC codes over a noisy channel'             
- gauss_elim.py      
Original code for fast gaussian elimination on GF2 (binary matrix)          
- formatter.py           
Make standard H matrix format (diagonal on parity part)         
- submatrix_sampler.py             
Sample row and column          
- dubiner_sparsifier.py         
Sparsify a binary matrix       
- bit_flip_decoder_sequential.py         
Decode using hard decision (no soft information)         
- block_recover.py          
Generate all cyclic shifts of a QC-LDPC parity check vector              
- QCLDPC_sampler.py       
Sample QC LDPC H matrix and codewords       
- variables.py          
Contains important parameters             
- verifier.py                
Format H matrix into diagonal format and verify if it is same as the original one  
- util.py          
Save matrix into image file and data into text/csv file


## version history     
2025.09.06 Selected necessary functions from prior project LDPC          
2025.09.07 Revised functions and conducted QC LDPC testing              


## License
Available for non-commercial use      