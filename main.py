from scipy.io import loadmat
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mne.io import RawArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import preprocessing
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from mne.decoding import CSP
from sklearn.svm import SVC
from DL_main import DL_main
from scipy import interp
from scipy.signal import welch
from sklearn.metrics import roc_curve, auc

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='ECG_reconstruction')
    parser.add_argument("--seed", default=1657, help="Seed number")
    parser.add_argument("--batch_size", default=64, help="Batch_size")
    parser.add_argument("--data_path", default='D:\Memory-Data\BCI_data\Scalp\S12filt.mat', help="Data_path")
    parser.add_argument("--k_fold", default=10, help="k_fold")
    parser.add_argument("--n_epochs", default=50, help="k_fold")
    args = parser.parse_args()


    np.random.seed(seed=arfs.seed)

    Data = loadmat(args.data_path)
    signal=np.array(Data['trial'])

    X_pre= signal[0:250,:,:,:]
    X_on=signal[250:500,:,:,:]

    X= X_on.transpose(3, 1, 2,0)
    X_size=np.shape(X)

    Y=np.reshape(np.array(Data['Label']),-1)
    Y_reshape=Y[:].T

    cv = StratifiedKFold(n_splits=args.k_fold, shuffle = True)

    #cv = KFold(n_splits=10 )
    model_SVM = SVC(kernel='linear', probability=True)
    model_LDA = LinearDiscriminantAnalysis()
    model_rf= ExtraTreesClassifier(n_estimators=500, random_state=0)

    csp = CSP(n_components=3, reg=None, log=True, norm_trace=False)
    clf_LDA = Pipeline([('CSP', csp), ('LDA', model_LDA)])
    clf_SVM = Pipeline([('CSP', csp), ('SVM', model_SVM)])
    clf_rf = Pipeline([('CSP', csp), ('RF', model_rf)])


    score_LDA=[]
    score_SVM=[]
    score_rf=[]
    score_DL=[]
    CSP_train=[]
    CSP_test=[]
    freq_band=4
    CSP_train_band= None
    CSP_test_band= None

    for train, test in cv.split(X, Y_reshape):
        a=X[train]
        a_size = np.shape(a)
        b=X[test]
        b_size = np.shape(b)
        train_band=np.reshape(a[:,:,1,:],(a_size[0],a_size[1],a_size[3]))
        test_band = np.reshape(b[:, :, 1, :], (b_size[0], b_size[1], b_size[3]))
        CSP_train_band=csp.fit_transform(train_band, Y_reshape[train])
        CSP_train =  CSP_train_band
        CSP_test_band = csp.transform (test_band)
        CSP_test=CSP_test_band

        for band in range(3):
            train_band = np.reshape(a[:, :, band+1, :], (a_size[0], a_size[1], a_size[3]))
            test_band = np.reshape(b[:, :, band+1, :], (b_size[0], b_size[1], b_size[3]))
            CSP_train_band = csp.fit_transform(train_band, Y_reshape[train])
            CSP_train = np.concatenate(([CSP_train, CSP_train_band]), axis=1)
            CSP_test_band = csp.transform(test_band)
            CSP_test = np.concatenate(([CSP_test, CSP_test_band]), axis=1)


        model_LDA.fit(CSP_train,Y_reshape[train])
        score_LDA.append(model_LDA.score(CSP_test, Y_reshape[test]))

        model_SVM.fit(CSP_train, Y_reshape[train])
        score_SVM.append(model_SVM.score(CSP_test, Y_reshape[test]))

        model_rf.fit(CSP_train, Y_reshape[train])
        score_rf.append(model_rf.score(CSP_test, Y_reshape[test]))

        X_train = np.reshape(a, (a_size[0], a_size[2], a_size[1], a_size[3]))
        X_test = np.reshape(b, (b_size[0], b_size[2], b_size[1], b_size[3]))
        score_DL.append(DL_main(X_train, X_test, Y_reshape[train], Y_reshape[test], args))


    mean_auc_LDA = np.mean(score_LDA)
    std_auc_LDA = np.std(score_LDA)
    print (score_LDA)
    print ( "LDA",mean_auc_LDA ,std_auc_LDA )

    mean_auc_SVM = np.mean(score_SVM)
    std_auc_SVM = np.std(score_SVM)
    print (score_SVM)
    print ( "SVM",mean_auc_SVM ,std_auc_SVM )

    mean_auc_rf = np.mean(score_rf)
    std_auc_rf = np.std(score_rf)
    print(score_rf)
    print ( "RF",mean_auc_rf ,std_auc_rf )

    mean_auc_dl = np.mean(score_DL)
    std_auc_dl = np.std(score_DL)
    print(score_DL)
    print ( "DL",mean_auc_dl ,std_auc_dl )
