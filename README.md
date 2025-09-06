# QC LDPC recovery
This project aims to recover H matrix of a LDPC code without candidate set on a noisy channel      

Final goal is to fully recover Parity Check Matrix(H) of a NAND flash memory      

[Theoretical background][link]

[link]:https://bluesparrow2000.github.io/paperreview/LDPC/


## Installations
This project is based on python. Below are the packages that needs to be installed:

numpy                     
numba                     
scipy                   
matplotlib                   

## Files
- main.py
An executable file based on paper 'progressive reconstruction of LDPC H matrix ...'
- gauss_elim.py      
Original code for fast gaussian elimination on GF2 (binary matrix)
- extracter.py       
Extracts parity check vector from ECO matrix                  
- verifier.py               
Format H matrix into diagonal format and verify if it is same as the original one     
- dubiner_sparsifier.py        
Sparsify a binary matrix       

## version history
2025.09.07 Collected functions that are necessary for QC LDPC testing


## License
Available for non-commercial use      