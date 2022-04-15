import warnings
warnings.filterwarnings(action='ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OrdinalEncoder
from sklearn.model_selection import cross_validate

#########################################################################################################
# 공통 변수
#########################################################################################################
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
reg_estimators = [
    LinearRegression(fit_intercept=True),
    Lasso(alpha=0, max_iter=1000),
    Ridge(alpha=1, max_iter=1000),
    MLPRegressor(max_iter=500, alpha = 0.1, verbose = False, early_stopping = True, hidden_layer_sizes = (100, 10)),
    KNeighborsRegressor(n_neighbors = 5),
    SVR(),
    DecisionTreeRegressor(max_depth=5, criterion='entropy'),
    RandomForestRegressor(n_estimators=1000, max_depth=5, criterion='entropy'),
    GradientBoostingRegressor(n_estimators=1000, loss='deviance'),
    XGBRegressor(n_estimators=1000, eval_metric='merror'),
    LGBMRegressor(n_estimators=1000),
    ExtraTreesRegressor(n_estimators=1000, criterion='entropy'),
    AdaBoostRegressor(),
    SGDRegressor(penalty='l1', max_iter = 50, alpha = 0.001, early_stopping = True, n_iter_no_change = 3),
]

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
cls_estimators = [
    LogisticRegression(max_iter=100, penalty='l2', C=1, solver='saga'),
    MLPClassifier(max_iter=500, alpha = 0.1, verbose = False, early_stopping = True, hidden_layer_sizes = (100, 10)),
    KNeighborsClassifier(n_neighbors=5),
    SVC(gamma='auto'),
    SVC(kernel="linear", C=0.025),
    DecisionTreeClassifier(max_depth=5, criterion='entropy'),
    RandomForestClassifier(n_estimators=1000, max_depth=5, criterion='entropy'),
    GradientBoostingClassifier(n_estimators=1000, loss='deviance'),
    XGBClassifier(n_estimators=1000, eval_metric='merror'),
    LGBMClassifier(n_estimators=1000),
    ExtraTreesClassifier(n_estimators=1000, criterion='entropy'),
    AdaBoostClassifier(),
    HistGradientBoostingClassifier(),
    QuadraticDiscriminantAnalysis(),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    GaussianNB()
]

cls_bi_score = ['accuracy', 'precision', 'recall', 'f1']
cls_mt_score = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'precision_micro', 'recall_micro', 'f1_micro' ]

palette = 'Set1'
colors = sns.color_palette(palette)
markers = ['o', 's', '^', 'x']

#########################################################################################################
# 데이터 불러오기 - load
######################################################################################################### 
from sklearn.datasets import load_boston, load_breast_cancer, load_iris
def load_data(name):
    if name == 'boston':
        boston = load_boston()
        df_boston = pd.DataFrame(data=boston.data, columns=boston.feature_names)
        df_boston['target'] = boston.target
        df = df_boston
    elif name == 'cancer':
        cancer = load_breast_cancer()
        df_cancer = pd.DataFrame(data=cancer.data, columns=cancer.feature_names)
        df_cancer['target'] = cancer.target
        df = df_cancer
    elif name == 'iris':
        iris = load_iris()
        df_iris = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        df_iris['target'] = iris.target
        df = df_iris
    else:
        print('Support Data = boston, cancer, iris')
        return
    X, y = df.iloc[:,:-1], df.iloc[:,-1]
    return df, X, y.values


#########################################################################################################
# 데이터 탐색 - eda
#########################################################################################################
def eda_features(df, round=3, sort=False, plot=False):
    rtn = pd.DataFrame(
        data={
            'dtypes':df.dtypes.values,
            'count':np.round(df.count().values, round),
            'nunique':np.round(df.nunique().values, round),
            'nduplicate':np.round(df.count().values-df.nunique().values, round),
            'na':np.round(df.isna().sum().values, round)},
        index = df.columns)
    if sort:
        rtn = rtn.sort_values('dtypes')
    if plot:
        rtn.plot.bar(figsize=(12, 5), cmap=plt.cm.Set2)
        plt.xticks(rotation=90)
        plt.show()
    return rtn

def eda_range(df, round=3, sort=False, plot=False):
    rtn = pd.DataFrame(
        data={
            'dtypes':df.dtypes.values},
        index = df.columns)
    for c in df.columns:
        isnum = df[c].dtypes not in [ 'object', 'category' ]
        Q3, Q1 = df[c].quantile([.75, .25]) if isnum else [0, 0]
        rtn.loc[c, 'mean'] = np.round(df[c].mean(), round) if isnum else ''
        rtn.loc[c, 'std'] = np.round(df[c].std(), round) if isnum else ''
        rtn.loc[c, 'max'] = np.round(df[c].max(), round) if isnum else ''
        rtn.loc[c, 'Q3'] = np.round(Q3, round) if isnum else ''
        rtn.loc[c, 'Q2'] = np.round(df[c].median(), round) if isnum else ''
        rtn.loc[c, 'Q1'] = np.round(Q1, round) if isnum else ''
        rtn.loc[c, 'min'] = np.round(df[c].min(), round) if isnum else ''
    if sort:
        rtn = rtn.sort_values('dtypes')
    if plot:
        ncol = 5
        nrow = int(len(df.columns)/ncol+1)
        df.plot(kind='box', subplots=True, layout=(nrow, ncol), figsize=(12, 3*nrow))
        plt.tight_layout()
        plt.show()
    return rtn

def eda_na(df, round=3, sort=False, plot=False):
    rtn = pd.DataFrame(
        data={
            'dtypes':df.dtypes.values,
            'count':np.round(df.count().values, round),
            'na':np.round(df.isna().sum().values, round)},
        index = df.columns)
    rtn['na(%)'] = np.round(rtn['na']/df.shape[0]*100, round)
    if sort:
        rtn = rtn.sort_values('na', ascending=False)
    if plot:
        plt.figure(figsize=(12, 5))
        plt.bar(rtn.index, rtn.na, color='#a1c9f4')
        for i in range(rtn.shape[0]):
            if rtn.iloc[i, 2] > 0:
                plt.text(i, rtn.iloc[i, 2], f'{rtn.iloc[i, 2]:.0f}', ha='center', color='r')
        plt.xticks(rotation=90)
        plt.show()
    return rtn

def eda_outlier(df, round=3, sort=False, plot=False):
    rtn = pd.DataFrame(
        data={
            'dtypes':df.dtypes.values,
            'count':df.count().values},
        index = df.columns)
    for c in df.columns:
        isnum = df[c].dtypes not in [ 'object', 'category' ]
        Q3, Q1 = df[c].quantile([.75, .25]) if isnum else [0, 0]
        UL, LL = Q3+1.5*(Q3-Q1), Q1-1.5*(Q3-Q1)
        rtn.loc[c, 'noutlier'] = df.loc[(df[c] < LL) | (df[c] > UL), c].count() if isnum else 0
        rtn.loc[c, 'noutlier(%)'] = np.round((df.loc[(df[c]<LL)|(df[c]>UL), c].count()/df[c].count())*100, round)  if isnum else ''
        rtn.loc[c, 'top'] = np.round(Q3+1.5*(Q3-Q1), round) if isnum else ''
        rtn.loc[c, 'bottom'] = np.round(Q1-1.5*(Q3-Q1), round) if isnum else ''
        rtn.loc[c, 'ntop'] = np.sum(df[c] > UL) if isnum else ''
        rtn.loc[c, 'nbottom'] = np.sum(df[c] < LL) if isnum else ''
    if sort:
        rtn = rtn.sort_values('noutlier', ascending=False)
    if plot:
        plt.figure(figsize=(12, 5))
        plt.bar(rtn.index, rtn.noutlier, color='#a1c9f4')
        for i in range(rtn.shape[0]):
            if rtn.iloc[i, 2] > 0:
                plt.text(i, rtn.iloc[i, 2], f'{rtn.iloc[i, 2]:.0f}', ha='center', color='r')
        plt.xticks(rotation=90)
        plt.show()
    return rtn


def eda_cls_count(y, round=3, sort=False, plot=False, ax=None, title=None):
    cat = pd.Series(data=y)
    tmp = cat.value_counts()
    if plot :
        if ax == None:
            ax = plt
            plt.title(title)
        else:
            ax.set_title(title)
        ax.pie(x = tmp.values, labels = tmp.index, colors = colors, autopct = '%.2g%%', startangle = 90, wedgeprops={'width':0.8})
        
    rtn = pd.DataFrame(data=tmp.values, index=tmp.index, columns=['count'])
    rtn['count%'] = np.round(rtn['count'] / cat.shape[0] * 100, round)
    return rtn.sort_index() if sort else rtn


import scipy.stats as stats
from sklearn.preprocessing import OrdinalEncoder
def eda_corr(df, target, round=3, sort=False, plot=False):
    tmp = df[df[target].notna()].copy()
    for c in df.columns:
        if tmp[c].dtypes in [ 'object', 'category' ]:
            tmp[c] = OrdinalEncoder().fit_transform(tmp[[c]])
    rtn = pd.DataFrame(data={'dtypes':df.dtypes.values}, index = df.columns)
    target_type = 'num' if df[target].dtypes not in [ 'object', 'category' ] else 'cat'
    for c in tmp.columns:
        tmp_c = tmp[tmp[c].notna()] 
        c_type = 'num' if df[c].dtypes not in [ 'object', 'category' ] else 'cat'
        if target_type == 'num' and target_type == c_type:
            rtn.loc[c, 'pearsonr'] = np.round(stats.pearsonr(tmp_c[target], tmp_c[c])[0], round)
        else:
            rtn.loc[c, 'pearsonr'] = ''
        rtn.loc[c, 'spearmanr'] = np.round(stats.spearmanr(tmp_c[target], tmp_c[c])[0], round)
        rtn.loc[c, 'kendalltau'] = np.round(stats.kendalltau(tmp_c[target], tmp_c[c])[0], round)
    if sort:
        rtn = rtn.sort_values('spearmanr', ascending=False)
    if plot:
        rtn.plot.bar(figsize=(12, 5), cmap=plt.cm.Set2)
        plt.xticks(rotation=90)
        plt.show()
    return rtn


def eda_hist(df, target, features):
    ncol = 5 if len(features) > 5 else len(features)
    nrow = int(len(features)/ncol+1)
    fig, axs = plt.subplots(nrow, ncol, figsize=(12, 2.5*nrow), sharey=True)
    sns.set(font_scale = 0.8)
    for i, f in enumerate(features):
        p = sns.histplot(
            data = df,
            x = f,
            hue = target,
            palette = 'Set2',
            bins = 10,
            multiple = 'stack',
            stat = 'count',
            ax = axs[int(i/5), i%5])
        p.set_xlabel(f, fontsize = 10)
        if i>0:
            p.legend([],[], frameon=False)
    plt.tight_layout()

def eda_feature_importance(X, y, type='reg', columns=None, sort=False, plot=False):
    model1 = DecisionTreeClassifier().fit(X, y) if type=='cls' else DecisionTreeRegressor().fit(X, y)
    model2 = RandomForestClassifier().fit(X, y) if type=='cls' else RandomForestRegressor().fit(X, y)
    model3 = XGBClassifier(eval_metric='merror').fit(X, y) if type=='cls' else XGBRFRegressor(eval_metric='merror').fit(X, y)
    model4 = LGBMClassifier().fit(X, y) if type=='cls' else LGBMRegressor().fit(X, y)
    rtn = pd.DataFrame(
        data={
            model1.__class__.__name__:pre_scale(model1.feature_importances_, 'minmax'),
            model2.__class__.__name__:pre_scale(model2.feature_importances_, 'minmax'),
            model3.__class__.__name__:pre_scale(model3.feature_importances_, 'minmax'),
            model4.__class__.__name__:pre_scale(model4.feature_importances_, 'minmax')},
        index = columns)
    rtn['mean'] = rtn.mean(axis=1)
    if sort:
        rtn = rtn.sort_values('mean', ascending=False)
    if plot:
        rtn.iloc[:10,:-1].plot.bar(figsize=(12, 5), cmap=plt.cm.Set1)
        plt.xticks(rotation=90)
        plt.title('feature importance TOP 10')
        plt.show()
    return rtn


#########################################################################################################
# 통계 분석 - stat
#########################################################################################################
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.sandbox.stats.multicomp import MultiComparison
import scipy.stats as stats
def stat_anova_1way(df, groups, endog):
    model = ols(endog+'~ C('+groups+')', df).fit()
    al = anova_lm(model)
    print(al, '\n')
    if al.iloc[0, -1] < 0.05:
        tmp = df[[endog, groups]].dropna()
        comp = MultiComparison(tmp[endog], tmp[groups])
        result = comp.allpairtest(stats.ttest_ind, method='bonf')
        print(result[0])

def stat_chi2(df, group1, group2, round=3):
    contingency = pd.crosstab(df[group1], df[group2])
    _, pvalue, _, expected = stats.chi2_contingency(contingency, correction=False)  
    expected = np.round(pd.DataFrame(data=expected), round)
    expected.columns = contingency.columns
    contingency = np.round(contingency, 3)
    contingency['Type'] = '관측'
    expected['Type'] = '예측'
    c = pd.concat([contingency, expected])
    return pvalue, c.reset_index().rename(columns={'index':group1}).set_index(['Type', group1])

from statsmodels.stats.outliers_influence import variance_inflation_factor
def stat_vif(df, sort=False, plot=False):
    rtn = pd.DataFrame(
        data = [ variance_inflation_factor(df.values, i) for i in range(df.shape[1]) ],
        columns = ['VIF'],
        index = df.columns)
    if sort:
        rtn = rtn.sort_values('VIF', ascending=False)
    if plot:
        rtn.plot.bar(figsize=(12, 5), cmap=plt.cm.Set2)
        plt.xticks(rotation=90)
        plt.show()
    return rtn


#########################################################################################################
# 데이터 전처리 - pre
#########################################################################################################
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer, PolynomialFeatures
def pre_scale(data, method, degree=None, bias=True):
    scaled_data = np.array(data).reshape(-1, 1) if data.ndim == 1 else data
    if method == 'minmax':
        scaled_data = MinMaxScaler().fit_transform(scaled_data)
    elif method == 'standard':
        scaled_data = StandardScaler().fit_transform(scaled_data)
    elif method == 'power':
        scaled_data = PowerTransformer(method='yeo-johnson').fit_transform(scaled_data)
    elif method == 'log':
        scaled_data = np.log1p(np.array(scaled_data))
    if degree != None:
        scaled_data = PolynomialFeatures(degree=degree, include_bias=bias).fit_transform(scaled_data)
    return scaled_data.ravel() if data.ndim == 1 else scaled_data

def pre_round(df, round=3):
    rtn = df.copy()
    types = rtn.dtypes.values
    for i, c in enumerate(rtn.columns):
        if str(types[i]).find('float') >= 0:
            rtn[c] = np.round(df[c], round)
    rtn.fillna('', inplace=True)
    return rtn


#########################################################################################################
# 시각화 - plt
#########################################################################################################
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
def plt_decompress(X, y, method='svd', ax=None, alpha=0.7):
    if method == 'svd':
        comp = TruncatedSVD(n_components=2)
    elif method == 'tsne':
        comp = TSNE(n_components=2)
    else:
        comp = PCA(n_components=2)

    if ax == None:
        ax = plt
        plt.title(method)
    else:
        ax.set_title(method)

    groups = np.unique(y)
    X_comp = comp.fit_transform(X, y)
    for i, g in enumerate(groups):     
        idx = np.where(y==g)
        ax.scatter(X_comp[idx,0], X_comp[idx,1], s=30, color=colors[i], alpha=alpha)

def plt_hist(df, features, target=None):
    ncol = 5 if len(features) > 5 else len(features)
    nrow = 1 if ncol < 5 else int(np.ceil(len(features)/ncol))
    fig, ax = plt.subplots(nrow, ncol, figsize=(12, 3*nrow), sharey=True)
    sns.set(font_scale = 0.8)
    for i, f in enumerate(features):
        axs = ax[i%5] if nrow==1 else ax[int(i/5), i%5]
        if target==None:
            p = sns.histplot(data=df, x=f, palette=palette, bins=10, multiple ='stack', stat ='count', ax=axs)
        else:
            p = sns.histplot(data=df, x=f, hue=target, palette=palette, bins=10, multiple ='stack', stat ='count', ax=axs)
        p.set_xlabel(f, fontsize = 10)
        p.set(ylabel=None)
        if i > 0:
            p.legend([],[], frameon=False)
    plt.tight_layout()

def plt_kde(df, features):
    ncol = 5 if len(features) > 5 else len(features)
    nrow = 1 if ncol < 5 else int(np.ceil(len(features)/ncol))
    fig, ax = plt.subplots(nrow, ncol, figsize=(12, 3*nrow))
    sns.set(font_scale = 0.8)
    for i, f in enumerate(features):
        axs = ax[i%5] if nrow==1 else ax[int(i/5), i%5]
        p = sns.kdeplot(data=df, x=f, shade=True, ax=axs)
        p.set_xlabel(f, fontsize = 10)
        p.set(ylabel=None)
    plt.tight_layout()
    

#########################################################################################################
# 회귀분석 - reg
#########################################################################################################
def reg_equation(model, columns, name='model', round=3):
    equa = list(model.coef_)
    equa.insert(0, model.intercept_)
    cols = columns.insert(0, 'intercept')
    return pd.DataFrame(data=equa, columns=[name], index=cols)

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
def reg_poly_fit(X, y, degree, plot=False):
    xval = np.linspace(np.min(X), np.max(X), 100).reshape(-1, 1)
    yval = {}
    for n in range(1, degree+1):
        model = make_pipeline(
            PolynomialFeatures(degree=n),
            LinearRegression(fit_intercept=True)
        ).fit(X, y)
        yval['poly'+str(n)] = np.ravel(model.predict(xval))
    rtn = pd.DataFrame(data=yval, index=np.ravel(xval))
    if plot:
        plt.figure(figsize=(12, 5))
        color = sns.color_palette('Set2')
        plt.scatter(X, y, c='k', alpha=0.2, label='data')
        for i, c in enumerate(rtn.columns):
                plt.plot(rtn.index, rtn[c], color=color[i], lw=3, alpha=.7, label=c)
        plt.title(f'Reg Plot(Degree 1 ~ {degree})')
        plt.xlabel('Features')
        plt.ylabel('Target')
        plt.legend()
        plt.show()
    return rtn

def reg_poly_resid(X, y, degree, scoring='se', plot=False):
    models = []
    xval = X
    yval = {}
    for n in range(1, degree+1):
        model = make_pipeline(
            StandardScaler(),
            PolynomialFeatures(degree=n),
            LinearRegression(fit_intercept=True)
        ).fit(X, y)
        models.append(model)
        if scoring == 'ae':
            yval['poly'+str(n)] = np.abs(np.ravel(model.predict(xval) - y.values))
        elif scoring == 'se':
            yval['poly'+str(n)] = np.square(np.ravel(model.predict(xval) - y.values))
        else:
            yval['poly'+str(n)] = np.ravel(model.predict(xval) - y.values)
    rtn = pd.DataFrame(data=yval, index=np.ravel(xval)).sort_index(ascending=True)    
    if plot:
        plt.figure(figsize=(12, 5))
        color = sns.color_palette('Set2')
        for i, c in enumerate(rtn.columns):
            plt.bar(rtn.index, rtn[c]+(i*0.2), color=color[i], alpha=0.5, label=c, width=0.3)
        plt.title(f'Reg Error(Degree 1 ~ {degree})')
        plt.xlabel('Features')
        plt.ylabel('Residuals')
        plt.legend()
        plt.show()
    return rtn

#########################################################################################################
# 모델 탐색 - test
#########################################################################################################
def explore_reg_model(X, y, cv=10, verbose=True, sort=False, plot=False):
    models = reg_estimators
    results  = []
    for i, model in enumerate(models):
        if verbose:
            print(f'({i+1}/{len(models)}) reg_model_test: {model.__class__.__name__}\n{model.get_params()}')
        else:
            print(f'({i+1}/{len(models)}) reg_model_test: {model.__class__.__name__}')
        results.append(cv_reg_score(model, X, y, cv))
    rtn = pd.concat(results, axis=1).T
    if sort:
        rtn = rtn.sort_values('fit_Time')
    if plot:
        rtn.plot.bar(figsize=(15, 8), cmap=plt.cm.Set1)
        plt.show()
    return rtn


def explore_cls_model(X, y, cv=10, verbose=True, sort=False, plot=False):
    models = cls_estimators
    results  = []
    for i, model in enumerate(models):
        if verbose:
            print(f'({i+1}/{len(models)}) reg_model_test: {model.__class__.__name__}\n{model.get_params()}')
        else:
            print(f'({i+1}/{len(models)}) reg_model_test: {model.__class__.__name__}')
        results.append(cv_cls_score(model, X, y, cv))
    rtn = pd.concat(results, axis=1).T
    if sort:
        rtn = rtn.sort_values('fit_Time')
    if plot:
        rtn.plot.bar(figsize=(15, 8), cmap=plt.cm.Set1)
        plt.show()
    return rtn

#########################################################################################################
# 모형 시각화 - show
#########################################################################################################
def show_feature_importance(model, columns, sort=False, plot=False):
    rtn = pd.DataFrame(data=np.array(model.feature_importances_), index=columns, columns=['importances'])
    if sort:
        rtn = rtn.sort_values(by='importances', ascending=False)
    if plot:
        rtn.plot.bar(figsize=(12, 5), cmap=plt.cm.Set2)
        plt.show()
    return rtn

def show_decision_boundaries(X, y, decision_model, **model_params):
    '''
    input :  X : dataframe(over 2 Features, 2개 이상이면 앞쪽 2개 변수만 사용)
             y : target
    '''
    features = X.columns

    X = np.array(X)
    y = np.array(y).flatten()
    reduced_data = X[:, :2]
    model = decision_model(**model_params).fit(reduced_data, y)
    h = .02
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])    
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.Set2)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=15, alpha=0.5, cmap=plt.cm.Set1)
    plt.xlabel(features[0])
    plt.ylabel(features[0])
    return plt


#########################################################################################################
# 모델 검증 - cv
#########################################################################################################
from sklearn.model_selection import cross_validate
def cv_reg_score(model, X, y, cv=5, title=None):
    cv = cross_validate(
        model,
        X, 
        y,
        cv=cv,
        scoring=('neg_mean_absolute_error', 'neg_root_mean_squared_error', 'r2'),
        return_train_score=True
    )
    if title == None:
        title = model.__class__.__name__
    tmp = pd.DataFrame(cv).mean()
    rtn = pd.DataFrame(data={
        'fit_Time': tmp['fit_time'],
        'train_MAE': -tmp['train_neg_mean_absolute_error'],
        'train_RMSE': -tmp['train_neg_root_mean_squared_error'],
        'train_R2': tmp['train_r2'],
        'test_MAE': -tmp['test_neg_mean_absolute_error'],
        'test_RMSE': -tmp['test_neg_root_mean_squared_error'],
        'test_R2': tmp['test_r2']},
        index=[title]).T
    return rtn

from sklearn.model_selection import cross_validate
def cv_cls_score(model, X, y, cv=5, title=None):
    scoring = cls_mt_score if len(np.unique(y)) > 2 else cls_bi_score
    cv = cross_validate(
        model,
        X, 
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=True)
    tmp = pd.DataFrame(cv).mean()
    if title == None:
        title = model.__class__.__name__
    if len(np.unique(y)) > 2:
        rtn = pd.DataFrame(data={
            'fit_Time': tmp['fit_time'],
            'train_accuracy': tmp['train_accuracy'],
            'train_precision_macro': tmp['train_precision_macro'],
            'train_recall_macro': tmp['train_recall_macro'],
            'train_f1_macro': tmp['train_f1_macro'],
            'train_precision_micro': tmp['train_precision_micro'],
            'train_recall_micro': tmp['train_recall_micro'],
            'train_f1_micro': tmp['train_f1_micro'],
            'test_accuracy': tmp['test_accuracy'],
            'test_precision_macro': tmp['test_precision_macro'],
            'test_recall_macro': tmp['test_recall_macro'],
            'test_f1_macro': tmp['test_f1_macro'],
            'test_precision_micro': tmp['test_precision_micro'],
            'test_recall_micro': tmp['test_recall_micro'],
            'test_f1_micro': tmp['test_f1_micro'],
            },
            index=[title]).T
        return rtn
    else:
        rtn = pd.DataFrame(data={
            'fit_Time': tmp['fit_time'],
            'train_accuracy': tmp['train_accuracy'],
            'train_precision': tmp['train_precision'],
            'train_recall': tmp['train_recall'],
            'train_f1': tmp['train_f1'],
            'test_accuracy': tmp['test_accuracy'],
            'test_precision': tmp['test_precision'],
            'test_recall': tmp['test_recall'],
            'test_f1': tmp['test_f1'],
            },
            index=[title]).T
        return rtn


#########################################################################################################
# 모형 평가 - show
#########################################################################################################
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
def metric_reg_score(y_test, y_pred, title='model'):
    rtn = pd.DataFrame(
        data = [
            np.round(mean_absolute_error(y_test, y_pred), 3),
            np.round(mean_squared_error(y_test, y_pred), 3),
            np.round(np.sqrt(mean_squared_error(y_test, y_pred)), 3),
            np.round(r2_score(y_test, y_pred), 3)
            ],
        columns = [title],
        index=['mae', 'mse', 'rmse', 'r2']
    )
    return rtn

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
def metric_cls_score(y_test, y_pred, title='model'):
    scoring = cls_mt_score if y.nunique() > 2 else cls_bi_score
    if y_test.nunique() > 2:
        rtn = pd.DataFrame(data=[
            accuracy_score(y_test, y_pred),
            precision_score(y_test, y_pred, average='macro'),
            recall_score(y_test, y_pred, average='macro'),
            f1_score(y_test, y_pred, average='macro'),
            precision_score(y_test, y_pred, average='micro'),
            recall_score(y_test, y_pred, average='micro'),
            f1_score(y_test, y_pred, average='micro')],
        columns=[title],
        index = scoring)
    else:
        rtn = pd.DataFrame(data=[
            accuracy_score(y_test, y_pred),
            precision_score(y_test, y_pred),
            recall_score(y_test, y_pred),
            f1_score(y_test, y_pred)],
        columns=[title],
        index = scoring)
    return rtn

from sklearn.metrics import roc_auc_score, plot_roc_curve
def metric_cls_auc_score(model, X_test, y_test, round=3, plot=True):
    if y_test.nunique() > 2:
        auc_score = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr')
    else:
        auc_score = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
        if plot:
            fig, ax = plt.subplots(figsize=(12,5))
            plot_roc_curve(model, X_test, y_test, ax=ax)
            plt.show()
    return np.round(auc_score, round)