@REM "@" means don't display the commands that come after the "@".

@ECHO ******************
@ECHO Install spec1d python module and run nosetests
@ECHO by Rohan Walker, April 2013
@ECHO ******************
cd "C:\Users\Rohan Walker\repo\spec1d"
python setup.py install
python setup.py clean --all
@REM cd out of spec1d folder or nosetests will run on source code rather than installed pakcage (not exactly sure where nosetests searches first.  Drill down on currnet directory or what?)
cd c:\
nosetests spec1d -v
@pause