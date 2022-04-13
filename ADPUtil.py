import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def eda_features(df, round=3, sort=False, plot=False):
    rtn = pd.DataFrame(
        data={
            'dtypes':df.dtypes.values,
            'count':df.count().values,
            'nunique':df.nunique().values,
            'nduplicate':df.count().values-df.nunique().values,
            'na':df.isna().sum().values},
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
            'count':df.count().values,
            'na':df.isna().sum().values},
        index = df.columns)
    rtn['na(%)'] = np.round(rtn['na']/df.shape[0]*100, 2)
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
            rtn.loc[c, 'spearmanr'] = np.round(stats.spearmanr(tmp_c[target], tmp_c[c])[0], round)
            rtn.loc[c, 'kendalltau'] = np.round(stats.kendalltau(tmp_c[target], tmp_c[c])[0], round)
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

def eda_cat_count(cat_array, round=3, sort=False, plot=False):
    cat = pd.Series(data=cat_array)
    tmp = cat.value_counts()
    rtn = pd.DataFrame(data=tmp.values, index=tmp.index, columns=['count'])
    rtn['count%'] = np.round(rtn['count'] / cat.shape[0] * 100, round)
    if plot :
        plt.pie(
            x = tmp.values,
            labels = tmp.index,
            colors = sns.color_palette('Set2'), 
            autopct = '%.2f%%',
            startangle = 90,
            wedgeprops={'width':0.8}, # 도넛그래프
            )
    return rtn.sort_index() if sort else rtn

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
            ax = axs[int(i/5), i%5]
            )
        p.set_xlabel(f, fontsize = 10)
        if i>0:
            p.legend([],[], frameon=False)
    plt.tight_layout()
    
def eda_round(df, round=3):
    rtn = df.copy()
    types = rtn.dtypes.values
    for i, c in enumerate(rtn.columns):
        if str(types[i]).find('float') >= 0:
            rtn[c] = np.round(df[c], round)
    rtn.fillna('', inplace=True)
    return rtn

from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.sandbox.stats.multicomp import MultiComparison
import scipy.stats as stats
def eda_anova(df, groups, endog):
    model = ols(endog+'~ C('+groups+')', df).fit()
    al = anova_lm(model)
    print(al)
    if al.iloc[0, -1] < 0.05:
        tmp = df[[endog, groups]].dropna()
        comp = MultiComparison(df[endog], df[groups])
        result = comp.allpairtest(stats.ttest_ind, method='bonf')
        print('\n', result[0])

def eda_chi2(df, cat1, cat2, round=3):
    contingency = pd.crosstab(df[cat1], df[cat2])
    _, pvalue, _, expected = stats.chi2_contingency(contingency, correction=False)  
    expected = np.round(pd.DataFrame(data=expected), round)
    expected.columns = contingency.columns
    contingency = np.round(contingency, 3)
    contingency['Type'] = '관측'
    expected['Type'] = '예측'
    c = pd.concat([contingency, expected])
    return pvalue, c.reset_index().rename(columns={'index':cat1}).set_index(['Type', cat1])

def reg_equation(model, columns, name='model', round=3):
    equa = list(model.coef_)
    equa.insert(0, model.intercept_)
    cols = columns.insert(0, 'intercept')
    rtn = pd.DataFrame(
        data=equa, columns=[name], index=cols)
    return rtn

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
def reg_score(y_test, y_pred, name='model'):
    rtn = pd.DataFrame(
        data = [
            np.round(mean_absolute_error(y_test, y_pred), 3),
            np.round(mean_squared_error(y_test, y_pred), 3),
            np.round(np.sqrt(mean_squared_error(y_test, y_pred)), 3),
            np.round(r2_score(y_test, y_pred), 3)
            ],
        columns = [name],
        index=['mae', 'mse', 'rmse', 'r2']
    )
    return rtn

from statsmodels.stats.outliers_influence import variance_inflation_factor
def reg_vif(df, sort=False, plot=False):
    rtn = pd.DataFrame(
        data = [ variance_inflation_factor(df.values, i) for i in range(df.shape[1]) ],
        columns = ['VIF'],
        index = df.columns
        )
    if sort:
        rtn = rtn.sort_values('VIF', ascending=False)
    if plot:
        rtn.plot.bar(figsize=(12, 5), cmap=plt.cm.Set2)
        plt.xticks(rotation=90)
        plt.show()
    return rtn

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

from sklearn.preprocessing import StandardScaler
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
        color = sns.color_palette('Set1')
        for i, c in enumerate(rtn.columns):
            plt.bar(rtn.index, rtn[c], color=color[i], alpha=.6, width=.3, label=c)
        plt.title(f'Reg Error(Degree 1 ~ {degree})')
        plt.xlabel('Features')
        plt.ylabel('Residuals')
        plt.legend()
        plt.show()
    return rtn

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor
def reg_model_test(X, y, cv=10, verbose=True, sort=False, plot=False):
    models = {
        'LinearRegression' : LinearRegression(fit_intercept=True),
        'Lasso' : Lasso(alpha=0, max_iter=1000),
        'Ridge' : Ridge(alpha=1, max_iter=1000),
        'SGDRegressor' : SGDRegressor(penalty='l1', max_iter = 50, alpha = 0.001, early_stopping = True, n_iter_no_change = 3),
        'MLPRegressor' : MLPRegressor(max_iter=500, alpha = 0.1, verbose = False, early_stopping = True, hidden_layer_sizes = (100, 10)),
        'KNeighborsRegressor' : KNeighborsRegressor(n_neighbors = 5),
        'SVR' : SVR(),
        'DecisionTreeRegressor' : DecisionTreeRegressor(),
        'BaggingRegressor' : BaggingRegressor(),
        'RandomForestRegressor' : RandomForestRegressor(),
        'ExtraTreesRegressor' : ExtraTreesRegressor(),
        'AdaBoostRegressor' : AdaBoostRegressor(),
        'GradientBoostingRegressor' : GradientBoostingRegressor()}
    results  = []
    for name, model in models.items():
        if verbose:
            print(f'model_test : {name}')
        test = cross_validate(
            model,
            X,
            y,
            scoring=('neg_root_mean_squared_error', 'r2'),
            cv=cv,
            return_train_score=True,
            n_jobs=-1)
        test = pd.DataFrame(data=test)
        test['model'] = name
        results.append(test)
    rtn = pd.concat(results).groupby('model').mean()
    if sort:
        rtn = rtn.sort_values('test_score', ascending=False)
    if plot:
        rtn.plot.bar(figsize=(12, 5), cmap=plt.cm.Set2)
        plt.xticks(rotation=90)
        plt.show()
    return rtn

from sklearn.metrics import roc_auc_score, plot_roc_curve
def cat_roc_auc_score(model, X_test, y_test, round=3, plot=True):
    if y_test.nunique() > 2:
        auc_score = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr')
    else:
        auc_score = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
        if plot:
            fig, ax = plt.subplots(figsize=(12,5))
            plot_roc_curve(model, X_test, y_test, ax=ax)
            plt.show()
        
    return np.round(auc_score, round)

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
def cat_score(y_test, y_pred, name='model'):
    if y_test.nunique() > 2:
        rtn = pd.DataFrame(data=[
            accuracy_score(y_test, y_pred),
            precision_score(y_test, y_pred, average='macro'),
            recall_score(y_test, y_pred, average='macro'),
            f1_score(y_test, y_pred, average='macro'),
            precision_score(y_test, y_pred, average='micro'),
            recall_score(y_test, y_pred, average='micro'),
            f1_score(y_test, y_pred, average='micro')],
        columns=[name],
        index = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'precision_micro', 'recall_micro', 'f1_micro' ])
    else:
        rtn = pd.DataFrame(data=[
            accuracy_score(y_test, y_pred),
            precision_score(y_test, y_pred),
            recall_score(y_test, y_pred),
            f1_score(y_test, y_pred)],
        columns=[name],
        index = ['accuracy', 'precision', 'recall', 'f1' ])
    return rtn

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
def cat_model_test(X, y, cv=10, verbose=True, sort=False, plot=False):
    models = {
        'LogisticRegression' : LogisticRegression(max_iter=100, solver='saga'),
        'DecisionTreeClassifier' : DecisionTreeClassifier(max_depth=5, criterion='entropy'),
        'KNeighborsClassifier' : KNeighborsClassifier(n_neighbors=5),
        'SVC-1' : SVC(gamma='auto'),
        'SVC-2' : SVC(kernel="linear", C=0.025),
        'GaussianProcessClassifier' : GaussianProcessClassifier(1.0 * RBF(1.0)),
        'GaussianNB' : GaussianNB(),
        'MLPClassifier' : MLPClassifier(max_iter=500, alpha = 0.1, verbose = False, early_stopping = True, hidden_layer_sizes = (100, 10)),
        'RandomForestClassifier' : RandomForestClassifier(max_depth=5, criterion='entropy'),
        'ExtraTreesClassifier' : ExtraTreesClassifier(criterion='entropy'),
        'AdaBoostClassifier' : AdaBoostClassifier(),
        'GradientBoostingClassifier, ' : GradientBoostingClassifier(loss='deviance'),
        'HistGradientBoostingClassifier' : HistGradientBoostingClassifier(),
        'XGBClassifier' : XGBClassifier(eval_metric='merror'),
        'LGBMClassifier' : LGBMClassifier(),
        'QuadraticDiscriminantAnalysis' : QuadraticDiscriminantAnalysis()}
    results = []
    for name, model in models.items():
        if verbose:
            print(f'model_test : {name}')
        test = cross_validate(
            model,
            X,
            y,
            scoring=('accuracy', 'f1_macro'),
            cv=cv,
            return_train_score=True,
            n_jobs=-1)
        test = pd.DataFrame(data=test)
        test['model'] = name
        results.append(test)
    rtn = pd.concat(results).groupby('model').mean()
    if sort:
        rtn = rtn.sort_values('test_f1_macro', ascending=False)
    if plot:
        rtn.plot.bar(figsize=(12, 5), cmap=plt.cm.Set2)
        plt.show()
    return rtn

def feature_importance(model, columns, sort=False, plot=False):
    rtn = pd.DataFrame(data=np.array(model.feature_importances_), index=columns, columns=['importances'])
    if sort:
        rtn = rtn.sort_values(by='importances', ascending=False)
    if plot:
        rtn.plot.bar(figsize=(12, 5), cmap=plt.cm.Set2)
        plt.show()
    return rtn

def plot_decision_boundaries(X, y, decision_model, **model_params):
    X = np.array(X)
    y = np.array(y).flatten()
    reduced_data = X[:, :2]
    model = decision_model(**model_params).fit(reduced_data, y)
    h = .02     # point in the mesh [x_min, m_max]x[y_min, y_max].    
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])    
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.Greys)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=15, alpha=0.5, cmap=plt.cm.viridis)
    plt.xlabel("Feature-1")
    plt.ylabel("Feature-2")
    return plt