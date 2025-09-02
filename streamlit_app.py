{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "521dd688",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\ARMAN PC\\AppData\\Roaming\\Python\\Python38\\site-packages\\sklearn\\linear_model\\_logistic.py:460: ConvergenceWarning: lbfgs failed to converge (status=1):\n",
      "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.\n",
      "\n",
      "Increase the number of iterations (max_iter) or scale the data as shown in:\n",
      "    https://scikit-learn.org/stable/modules/preprocessing.html\n",
      "Please also refer to the documentation for alternative solver options:\n",
      "    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression\n",
      "  n_iter_i = _check_optimize_result(\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "فنی و حرفه ای\n",
      "1.0 2.0 3.0 4.0 5.0 1.0 2.0 3.0\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\ARMAN PC\\AppData\\Roaming\\Python\\Python38\\site-packages\\sklearn\\base.py:465: UserWarning: X does not have valid feature names, but LogisticRegression was fitted with feature names\n",
      "  warnings.warn(\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.model_selection import train_test_split\n",
    "import sklearn.metrics as sm\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "\n",
    "\n",
    "\n",
    "df=pd.read_csv(\"data.csv\")\n",
    "x=df.drop('Suggestedcourse',axis=1)\n",
    "y=df.Suggestedcourse\n",
    "X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2)\n",
    "Robat=LogisticRegression()\n",
    "Robat.fit(X_train,y_train)\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "karbar1=float(input(':داشتن علاقه به ریاضی از 1 تا 5'))\n",
    "karbar2=float(input(':داشتن علاقه به تجربی از 1 تا 5'))\n",
    "karbar3=float(input(':داشتن علاقه به انسانی از 1 تا 5'))\n",
    "karbar4=float(input(':داشتن علاقه به فنی از 1 تا 5'))\n",
    "karbar5=float(input(':جمع نمرات ریاضی در 3 سال تحصیلی از 10 تا 20'))\n",
    "karbar6=float(input(':جمع نمرات تجربی در 3 سال تحصیلی از 10 تا 20'))\n",
    "karbar7=float(input(':جمع نمرات مطالعات ، فارسی ، عربی در 3 سال تحصیلی از 10 تا 20'))\n",
    "karbar8=float(input(': جمع نمرات کاروفناوری در 3 سال تحصیلی از 10 تا 20'))\n",
    "Danesh_Amoze=np.array([[karbar1,karbar2,karbar3,karbar4,karbar5,karbar6,karbar7,karbar8]])\n",
    "out1=Robat.predict(Danesh_Amoze)\n",
    "if out1==0:\n",
    "    print('ریاضی')\n",
    "elif out1==1:\n",
    "    print('تجربی')\n",
    "elif out1==2:\n",
    "    print('انسانی')\n",
    "elif out1==3:\n",
    "    print('فنی و حرفه ای')\n",
    "\n",
    "   \n",
    "print(karbar1,karbar2,karbar3,karbar4,karbar5,karbar6,karbar7,karbar8)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "8d525e84",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([3, 3, 1, 0, 2, 2, 3, 2, 1, 2, 0, 0, 0, 1, 1, 3, 1, 3, 3, 0, 3, 2,\n",
       "       0, 0, 2, 2, 2, 0, 3, 0, 2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 3, 2,\n",
       "       3, 2, 3, 2, 0, 0, 0, 0, 1, 1, 1, 2, 3, 0, 3, 0, 2, 0, 3, 1, 3, 1,\n",
       "       1, 0, 0, 1, 3, 0, 2, 3, 2, 3, 1, 3, 1, 2], dtype=int64)"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Robat_pre=Robat.predict(X_test)\n",
    "Robat_pre"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "10dff4f3",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "3      2\n",
       "85     1\n",
       "244    2\n",
       "133    2\n",
       "45     1\n",
       "256    2\n",
       "186    3\n",
       "365    2\n",
       "363    1\n",
       "316    2\n",
       "290    2\n",
       "224    0\n",
       "339    0\n",
       "168    1\n",
       "31     1\n",
       "7      2\n",
       "157    2\n",
       "87     0\n",
       "72     0\n",
       "241    0\n",
       "393    3\n",
       "96     2\n",
       "129    0\n",
       "236    0\n",
       "21     2\n",
       "318    3\n",
       "65     2\n",
       "92     3\n",
       "202    1\n",
       "177    2\n",
       "91     2\n",
       "281    1\n",
       "360    0\n",
       "358    3\n",
       "86     1\n",
       "53     0\n",
       "99     0\n",
       "324    2\n",
       "2      0\n",
       "169    3\n",
       "248    1\n",
       "237    2\n",
       "220    1\n",
       "119    2\n",
       "347    3\n",
       "56     2\n",
       "210    3\n",
       "289    2\n",
       "76     3\n",
       "74     2\n",
       "183    1\n",
       "152    0\n",
       "284    0\n",
       "79     3\n",
       "190    1\n",
       "63     2\n",
       "254    2\n",
       "97     2\n",
       "22     3\n",
       "149    3\n",
       "Name: Suggestedcourse, dtype: int64"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_test.head(60)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "84781c5e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "3     -1\n",
       "85    -2\n",
       "244    1\n",
       "133    2\n",
       "45    -1\n",
       "      ..\n",
       "81    -1\n",
       "349    0\n",
       "234    0\n",
       "145    2\n",
       "354    0\n",
       "Name: Suggestedcourse, Length: 80, dtype: int64"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "er= y_test - Robat_pre\n",
    "er"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "299a04e1",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "42\n",
      "38\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "42"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "sahih=0\n",
    "qhalat=0\n",
    "for i in er:\n",
    "    if(i==0):\n",
    "        sahih=sahih+1\n",
    "    if(i!=0):\n",
    "        qhalat=qhalat+1\n",
    "print(sahih)\n",
    "print(qhalat)\n",
    "sahih_darsad=(sahih*100)/360\n",
    "sahih"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "b09c5f62",
   "metadata": {},
   "outputs": [
    {
     "ename": "FileNotFoundError",
     "evalue": "[Errno 2] No such file or directory: 'data.csv'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mFileNotFoundError\u001b[0m                         Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[6], line 9\u001b[0m\n\u001b[0;32m      6\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[38;5;21;01msklearn\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mpreprocessing\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[38;5;28;01mimport\u001b[39;00m StandardScaler\n\u001b[0;32m      8\u001b[0m \u001b[38;5;66;03m# --- خواندن دیتاست ---\u001b[39;00m\n\u001b[1;32m----> 9\u001b[0m df \u001b[38;5;241m=\u001b[39m \u001b[43mpd\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mread_csv\u001b[49m\u001b[43m(\u001b[49m\u001b[38;5;124;43m\"\u001b[39;49m\u001b[38;5;124;43mdata.csv\u001b[39;49m\u001b[38;5;124;43m\"\u001b[39;49m\u001b[43m)\u001b[49m\n\u001b[0;32m     10\u001b[0m x \u001b[38;5;241m=\u001b[39m df\u001b[38;5;241m.\u001b[39mdrop(\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mSuggestedcourse\u001b[39m\u001b[38;5;124m'\u001b[39m, axis\u001b[38;5;241m=\u001b[39m\u001b[38;5;241m1\u001b[39m)\n\u001b[0;32m     11\u001b[0m y \u001b[38;5;241m=\u001b[39m df\u001b[38;5;241m.\u001b[39mSuggestedcourse\n",
      "File \u001b[1;32ma:\\aio\\.venv\\lib\\site-packages\\pandas\\io\\parsers\\readers.py:912\u001b[0m, in \u001b[0;36mread_csv\u001b[1;34m(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)\u001b[0m\n\u001b[0;32m    899\u001b[0m kwds_defaults \u001b[38;5;241m=\u001b[39m _refine_defaults_read(\n\u001b[0;32m    900\u001b[0m     dialect,\n\u001b[0;32m    901\u001b[0m     delimiter,\n\u001b[1;32m   (...)\u001b[0m\n\u001b[0;32m    908\u001b[0m     dtype_backend\u001b[38;5;241m=\u001b[39mdtype_backend,\n\u001b[0;32m    909\u001b[0m )\n\u001b[0;32m    910\u001b[0m kwds\u001b[38;5;241m.\u001b[39mupdate(kwds_defaults)\n\u001b[1;32m--> 912\u001b[0m \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[43m_read\u001b[49m\u001b[43m(\u001b[49m\u001b[43mfilepath_or_buffer\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mkwds\u001b[49m\u001b[43m)\u001b[49m\n",
      "File \u001b[1;32ma:\\aio\\.venv\\lib\\site-packages\\pandas\\io\\parsers\\readers.py:577\u001b[0m, in \u001b[0;36m_read\u001b[1;34m(filepath_or_buffer, kwds)\u001b[0m\n\u001b[0;32m    574\u001b[0m _validate_names(kwds\u001b[38;5;241m.\u001b[39mget(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mnames\u001b[39m\u001b[38;5;124m\"\u001b[39m, \u001b[38;5;28;01mNone\u001b[39;00m))\n\u001b[0;32m    576\u001b[0m \u001b[38;5;66;03m# Create the parser.\u001b[39;00m\n\u001b[1;32m--> 577\u001b[0m parser \u001b[38;5;241m=\u001b[39m \u001b[43mTextFileReader\u001b[49m\u001b[43m(\u001b[49m\u001b[43mfilepath_or_buffer\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[43mkwds\u001b[49m\u001b[43m)\u001b[49m\n\u001b[0;32m    579\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m chunksize \u001b[38;5;129;01mor\u001b[39;00m iterator:\n\u001b[0;32m    580\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m parser\n",
      "File \u001b[1;32ma:\\aio\\.venv\\lib\\site-packages\\pandas\\io\\parsers\\readers.py:1407\u001b[0m, in \u001b[0;36mTextFileReader.__init__\u001b[1;34m(self, f, engine, **kwds)\u001b[0m\n\u001b[0;32m   1404\u001b[0m     \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39moptions[\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mhas_index_names\u001b[39m\u001b[38;5;124m\"\u001b[39m] \u001b[38;5;241m=\u001b[39m kwds[\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mhas_index_names\u001b[39m\u001b[38;5;124m\"\u001b[39m]\n\u001b[0;32m   1406\u001b[0m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mhandles: IOHandles \u001b[38;5;241m|\u001b[39m \u001b[38;5;28;01mNone\u001b[39;00m \u001b[38;5;241m=\u001b[39m \u001b[38;5;28;01mNone\u001b[39;00m\n\u001b[1;32m-> 1407\u001b[0m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_engine \u001b[38;5;241m=\u001b[39m \u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43m_make_engine\u001b[49m\u001b[43m(\u001b[49m\u001b[43mf\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mengine\u001b[49m\u001b[43m)\u001b[49m\n",
      "File \u001b[1;32ma:\\aio\\.venv\\lib\\site-packages\\pandas\\io\\parsers\\readers.py:1661\u001b[0m, in \u001b[0;36mTextFileReader._make_engine\u001b[1;34m(self, f, engine)\u001b[0m\n\u001b[0;32m   1659\u001b[0m     \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mb\u001b[39m\u001b[38;5;124m\"\u001b[39m \u001b[38;5;129;01mnot\u001b[39;00m \u001b[38;5;129;01min\u001b[39;00m mode:\n\u001b[0;32m   1660\u001b[0m         mode \u001b[38;5;241m+\u001b[39m\u001b[38;5;241m=\u001b[39m \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mb\u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[1;32m-> 1661\u001b[0m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mhandles \u001b[38;5;241m=\u001b[39m \u001b[43mget_handle\u001b[49m\u001b[43m(\u001b[49m\n\u001b[0;32m   1662\u001b[0m \u001b[43m    \u001b[49m\u001b[43mf\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m   1663\u001b[0m \u001b[43m    \u001b[49m\u001b[43mmode\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m   1664\u001b[0m \u001b[43m    \u001b[49m\u001b[43mencoding\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43moptions\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mget\u001b[49m\u001b[43m(\u001b[49m\u001b[38;5;124;43m\"\u001b[39;49m\u001b[38;5;124;43mencoding\u001b[39;49m\u001b[38;5;124;43m\"\u001b[39;49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;28;43;01mNone\u001b[39;49;00m\u001b[43m)\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m   1665\u001b[0m \u001b[43m    \u001b[49m\u001b[43mcompression\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43moptions\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mget\u001b[49m\u001b[43m(\u001b[49m\u001b[38;5;124;43m\"\u001b[39;49m\u001b[38;5;124;43mcompression\u001b[39;49m\u001b[38;5;124;43m\"\u001b[39;49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;28;43;01mNone\u001b[39;49;00m\u001b[43m)\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m   1666\u001b[0m \u001b[43m    \u001b[49m\u001b[43mmemory_map\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43moptions\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mget\u001b[49m\u001b[43m(\u001b[49m\u001b[38;5;124;43m\"\u001b[39;49m\u001b[38;5;124;43mmemory_map\u001b[39;49m\u001b[38;5;124;43m\"\u001b[39;49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;28;43;01mFalse\u001b[39;49;00m\u001b[43m)\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m   1667\u001b[0m \u001b[43m    \u001b[49m\u001b[43mis_text\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mis_text\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m   1668\u001b[0m \u001b[43m    \u001b[49m\u001b[43merrors\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43moptions\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mget\u001b[49m\u001b[43m(\u001b[49m\u001b[38;5;124;43m\"\u001b[39;49m\u001b[38;5;124;43mencoding_errors\u001b[39;49m\u001b[38;5;124;43m\"\u001b[39;49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;124;43m\"\u001b[39;49m\u001b[38;5;124;43mstrict\u001b[39;49m\u001b[38;5;124;43m\"\u001b[39;49m\u001b[43m)\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m   1669\u001b[0m \u001b[43m    \u001b[49m\u001b[43mstorage_options\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43moptions\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mget\u001b[49m\u001b[43m(\u001b[49m\u001b[38;5;124;43m\"\u001b[39;49m\u001b[38;5;124;43mstorage_options\u001b[39;49m\u001b[38;5;124;43m\"\u001b[39;49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;28;43;01mNone\u001b[39;49;00m\u001b[43m)\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m   1670\u001b[0m \u001b[43m\u001b[49m\u001b[43m)\u001b[49m\n\u001b[0;32m   1671\u001b[0m \u001b[38;5;28;01massert\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mhandles \u001b[38;5;129;01mis\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m \u001b[38;5;28;01mNone\u001b[39;00m\n\u001b[0;32m   1672\u001b[0m f \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mhandles\u001b[38;5;241m.\u001b[39mhandle\n",
      "File \u001b[1;32ma:\\aio\\.venv\\lib\\site-packages\\pandas\\io\\common.py:859\u001b[0m, in \u001b[0;36mget_handle\u001b[1;34m(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)\u001b[0m\n\u001b[0;32m    854\u001b[0m \u001b[38;5;28;01melif\u001b[39;00m \u001b[38;5;28misinstance\u001b[39m(handle, \u001b[38;5;28mstr\u001b[39m):\n\u001b[0;32m    855\u001b[0m     \u001b[38;5;66;03m# Check whether the filename is to be opened in binary mode.\u001b[39;00m\n\u001b[0;32m    856\u001b[0m     \u001b[38;5;66;03m# Binary mode does not support 'encoding' and 'newline'.\u001b[39;00m\n\u001b[0;32m    857\u001b[0m     \u001b[38;5;28;01mif\u001b[39;00m ioargs\u001b[38;5;241m.\u001b[39mencoding \u001b[38;5;129;01mand\u001b[39;00m \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mb\u001b[39m\u001b[38;5;124m\"\u001b[39m \u001b[38;5;129;01mnot\u001b[39;00m \u001b[38;5;129;01min\u001b[39;00m ioargs\u001b[38;5;241m.\u001b[39mmode:\n\u001b[0;32m    858\u001b[0m         \u001b[38;5;66;03m# Encoding\u001b[39;00m\n\u001b[1;32m--> 859\u001b[0m         handle \u001b[38;5;241m=\u001b[39m \u001b[38;5;28;43mopen\u001b[39;49m\u001b[43m(\u001b[49m\n\u001b[0;32m    860\u001b[0m \u001b[43m            \u001b[49m\u001b[43mhandle\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m    861\u001b[0m \u001b[43m            \u001b[49m\u001b[43mioargs\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mmode\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m    862\u001b[0m \u001b[43m            \u001b[49m\u001b[43mencoding\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mioargs\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mencoding\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m    863\u001b[0m \u001b[43m            \u001b[49m\u001b[43merrors\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43merrors\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m    864\u001b[0m \u001b[43m            \u001b[49m\u001b[43mnewline\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[38;5;124;43m\"\u001b[39;49m\u001b[38;5;124;43m\"\u001b[39;49m\u001b[43m,\u001b[49m\n\u001b[0;32m    865\u001b[0m \u001b[43m        \u001b[49m\u001b[43m)\u001b[49m\n\u001b[0;32m    866\u001b[0m     \u001b[38;5;28;01melse\u001b[39;00m:\n\u001b[0;32m    867\u001b[0m         \u001b[38;5;66;03m# Binary mode\u001b[39;00m\n\u001b[0;32m    868\u001b[0m         handle \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mopen\u001b[39m(handle, ioargs\u001b[38;5;241m.\u001b[39mmode)\n",
      "\u001b[1;31mFileNotFoundError\u001b[0m: [Errno 2] No such file or directory: 'data.csv'"
     ]
    }
   ],
   "source": [
    "import streamlit as st\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "\n",
    "# --- خواندن دیتاست ---\n",
    "df = pd.read_csv(\"data.csv\")\n",
    "x = df.drop('Suggestedcourse', axis=1)\n",
    "y = df.Suggestedcourse\n",
    "\n",
    "# --- تقسیم داده ---\n",
    "X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)\n",
    "\n",
    "# --- مقیاس‌دهی داده‌ها ---\n",
    "scaler = StandardScaler()\n",
    "X_train = scaler.fit_transform(X_train)\n",
    "X_test = scaler.transform(X_test)\n",
    "\n",
    "# --- تعریف و آموزش مدل ---\n",
    "Robat = LogisticRegression(max_iter=1000)\n",
    "Robat.fit(X_train, y_train)\n",
    "\n",
    "# --- رابط کاربری Streamlit ---\n",
    "st.title(\"📚 انتخاب رشته تحصیلی هوشمند\")\n",
    "\n",
    "karbar1 = st.slider(\"علاقه به ریاضی (1 تا 5)\", 1, 5, 3)\n",
    "karbar2 = st.slider(\"علاقه به تجربی (1 تا 5)\", 1, 5, 3)\n",
    "karbar3 = st.slider(\"علاقه به انسانی (1 تا 5)\", 1, 5, 3)\n",
    "karbar4 = st.slider(\"علاقه به فنی (1 تا 5)\", 1, 5, 3)\n",
    "karbar5 = st.slider(\"مجموع نمرات ریاضی (10 تا 20)\", 10, 20, 15)\n",
    "karbar6 = st.slider(\"مجموع نمرات تجربی (10 تا 20)\", 10, 20, 15)\n",
    "karbar7 = st.slider(\"مجموع نمرات انسانی (10 تا 20)\", 10, 20, 15)\n",
    "karbar8 = st.slider(\"مجموع نمرات فنی (10 تا 20)\", 10, 20, 15)\n",
    "\n",
    "# --- آماده‌سازی ورودی کاربر ---\n",
    "Danesh_Amoze = np.array([[karbar1,karbar2,karbar3,karbar4,\n",
    "                          karbar5,karbar6,karbar7,karbar8]])\n",
    "\n",
    "Danesh_Amoze = scaler.transform(Danesh_Amoze)  # مقیاس‌دهی ورودی جدید\n",
    "\n",
    "# --- پیش‌بینی ---\n",
    "out1 = Robat.predict(Danesh_Amoze)[0]\n",
    "\n",
    "# --- نمایش نتیجه ---\n",
    "st.subheader(\"🔮 نتیجه پیشنهادی:\")\n",
    "if out1 == 0:\n",
    "    st.success(\"📘 رشته ریاضی\")\n",
    "elif out1 == 1:\n",
    "    st.success(\"🔬 رشته تجربی\")\n",
    "elif out1 == 2:\n",
    "    st.success(\"📖 رشته انسانی\")\n",
    "elif out1 == 3:\n",
    "    st.success(\"⚙️ رشته فنی و حرفه‌ای\")\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "267d46bd",
   "metadata": {},
   "outputs": [
    {
     "ename": "SyntaxError",
     "evalue": "invalid syntax (1722541631.py, line 1)",
     "output_type": "error",
     "traceback": [
      "\u001b[1;36m  Cell \u001b[1;32mIn[5], line 1\u001b[1;36m\u001b[0m\n\u001b[1;33m    streamlit run \"import streamlit as st.py\"\u001b[0m\n\u001b[1;37m              ^\u001b[0m\n\u001b[1;31mSyntaxError\u001b[0m\u001b[1;31m:\u001b[0m invalid syntax\n"
     ]
    }
   ],
   "source": [
    "streamlit run \"import streamlit as st.py\"\n",
    "\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": ".venv (3.8.8)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
