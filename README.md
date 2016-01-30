# ASE_extn

Q. How do I run the script?\
A. Execute com.ase.extn.cart.cart.py

Q. What parameters do I need to set in the script?\
A. Edit com.ase.extn.cart.cart.py to modify following variables -

	- strategy = 'projective'|'progressive'
	- system = 'all'|'apache'|'bc'|'bj'|'llvm'|'sqlite'|'x264'

Q. How can I interpret the results?\
A. For progressive sampling, individual output files are created under data/ouput for each system. These files contain a mapping between sample size and fault rate.
   For projective sampling, results are displayed in the console in the following format:
	
	System-id : <system under execution>
	Size of lambda set: <size of the data points used for the projected learning curve>
	------------------------------------------------------------------
	{<projected curve> : [<correlation value>, <optimal sample size>]}


Q. Where do I feed in the input files?\
A. Input files are read from data/input

Q. Where can I get further details?\
A. http://gsd.uwaterloo.ca/sites/default/files/PID3840471.pdf