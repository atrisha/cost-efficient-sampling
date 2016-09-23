# ASE_extn

Q. Where do I feed in the input files?<br />
A. Input files are read from com.ase.extn.constants.data.input

Q: How can I execute progressive sampling?<br />
A: Set the following parameters in com.ase.extn.constants.configs.py -
	-strategy = ’progressive’
	-system = ’apache’|’bc’|’bj’|’llvm’|’x264’|’sqlite’|’all’
To display figures : plot = True
To display learning curve: plot real cost = False
To display cost curve: plot real cost = True
Execute com.ase.extn.cart.base.py
Results are generated under com.ase.extn.constants.data.output

Q: How can I execute projective sampling?<br />
A: Set the following parameters in com.ase.extn.constants.configs.py
	-strategy = ’projective’
	-system = ’apache’|’bc’|’bj’|’llvm’|’x264’|’sqlite’|’all’
	-print detail = True
To display figures : plot = True
Execute com.ase.extn.cart.base.py
Results are displayed in the console

Q. How to setup the parameters for projective sampling?<br />
A.
Cost-ratio (R) : r = <value>
Feature-frequency threshold (thresh freq) : projective feature threshold = <value>
Multiplier for training-testing set split (θ) : th = <value>

Q. How to run t-way sampling?<br />
A. Set the following parameters in com.ase.extn.constants.configs.py
	-tway = 2 | 3
Execute com.ase.extn.tway.twaysample.py
Results are displayed in the console

Q. How to run sensitivity analysis?<br />
A. Set the following parameters in com.ase.extn.sensitivity.sanalysis.py
	-sensitivity = ’r’ | ’th’
	-com.ase.extn.constants.configs.r 0 to 1 = True : For SA in the interval [0,1] for R


Q. Where can I get further details?<br />
A. https://uwspace.uwaterloo.ca/bitstream/handle/10012/10406/Sarkar_Atri.pdf?sequence=3
