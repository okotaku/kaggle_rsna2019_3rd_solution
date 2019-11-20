mkdir exp/models
mkdir input
mkdir output
cd preprocess
python creating_demo.py
python make_user_df.py
cd ..

cd prediction
python exp10.py
python exp16.py
python exp17.py
python exp18.py
python exp19.py
python exp21.py
python exp22.py
python exp23.py
python exp24.py
cd ..

cd stack
python make_user_stackingdata_demo.py
python user_stacking_demo.py
