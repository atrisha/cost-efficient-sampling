# ASE_extn

Q. Where do I feed in the input files?<br />
A. Input files are read from com.ase.extn.constants.data.input

Q: How can I execute progressive sampling?<br />
A: Set the following parameters in com.ase.extn.constants.configs.py -<br />
	-strategy = ’progressive’<br />
	-system = ’apache’|’bc’|’bj’|’llvm’|’x264’|’sqlite’|’all’<br />
To display figures : plot = True<br />
To display learning curve: plot real cost = False<br />
To display cost curve: plot real cost = True<br />
Execute com.ase.extn.cart.base.py<br />
Results are generated under com.ase.extn.constants.data.output<br />

Q: How can I execute projective sampling?<br />
A: Set the following parameters in com.ase.extn.constants.configs.py<br />
	-strategy = ’projective’<br />
	-system = ’apache’|’bc’|’bj’|’llvm’|’x264’|’sqlite’|’all’<br />
	-print detail = True<br />
To display figures : plot = True<br />
Execute com.ase.extn.cart.base.py<br />
Results are displayed in the console<br />

Q. How to setup the parameters for projective sampling?<br />
A.
Cost-ratio (R) : r = [value] <br />
Feature-frequency threshold (thresh freq) : projective feature threshold = [value] <br />
Multiplier for training-testing set split (θ) : th = [value] <br />

Q. How to run t-way sampling?<br />
A. Set the following parameters in com.ase.extn.constants.configs.py<br />
	-tway = 2 | 3<br />
Execute com.ase.extn.tway.twaysample.py<br />
Results are displayed in the console<br />

Q. How to run sensitivity analysis?<br />
A. Set the following parameters in com.ase.extn.sensitivity.sanalysis.py<br />
	-sensitivity = ’r’ | ’th’<br />
	-com.ase.extn.constants.configs.r 0 to 1 = True : For SA in the interval [0,1] for R<br />


Q. Where can I get further details?<br />
A. https://uwspace.uwaterloo.ca/bitstream/handle/10012/10406/Sarkar_Atri.pdf?sequence=3<br />
