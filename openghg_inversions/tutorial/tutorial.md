# Very basic inversions tutorial

After cloning OpenGHG inversions, say to `Documents`, create a folder called `inversions_tutorial`. 

Assuming `openghg_inversions` is in the same folder (e.g. `Documents`), activate your environment and use the following command to create an object store with some data used to test inversions:
```
python openghg_inversions/scripts/make_test_store "inversions_tutorial_store"
```
This will make a folder called `inversions_tutorial_store` in the current directory (e.g. `Documents`), which is an OpenGHG object store with a small amount of data for running inversions.

In `openghg_inversions/tutorials` there is a sample `ini` file called `mhd_tac_ch4.ini`. This contains specifications for getting CH4 data at the stations MHD and TAC, along with a prior flux and boundary conditions, as well as for running the inversion.

To run the inversion, move to your `inversions_tutorial` directory and use the following command:
```
python ../openghg_inversions/openghg_inversions/hbmcmc/run_hbmcmc.py 2019-01-01 2019-02-01 --output-path $(pwd)
```
If `openghg_inversions` is not one level up from `inversions_tutorial`, you might need to change the path.

This will run an inversion!
