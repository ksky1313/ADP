# 데이터 변수별 컬럼타입과 na, unique 갯수 체크
def eda_features(df, round=3):
    rtn = pd.DataFrame(
        data={
            'dtypes':df.dtypes.values,
            'count':df.count().values,
            'nunique':df.nunique().values,
            'nduplicate':df.count().values-df.nunique().values,
            'na':df.isna().sum().values,
        },
        index = df.columns
    )
    return rtn

# 데이터 변수별 범위를 조회
def eda_range(df, round=3):
    rtn = pd.DataFrame(
        data={
            'dtypes':df.dtypes.values
        },
        index = df.columns
    )
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
    return rtn


# 데이터 변수별 na와 %를 조회
def eda_na(df, sort=True, round=3):
    rtn = pd.DataFrame(
        data={
            'dtypes':df.dtypes.values,
            'count':df.count().values,
            'na':df.isna().sum().values,
        },
        index = df.columns
    )
    rtn['na(%)'] = np.round(rtn['na']/df.shape[0]*100, 2)
    return rtn.sort_values('na', ascending=False) if sort else rtn

def eda_outlier(df, sort=True, round=3):
    rtn = pd.DataFrame(
        data={
            'dtypes':df.dtypes.values,
            'count':df.count().values,
        },
        index = df.columns
    )
    for c in df.columns:
        isnum = df[c].dtypes not in [ 'object', 'category' ]
        Q3, Q1 = df[c].quantile([.75, .25]) if isnum else [0, 0]
        UL, LL = Q3+1.5*(Q3-Q1), Q1-1.5*(Q3-Q1)
        rtn.loc[c, 'noutlier'] = df.loc[(df[c] < LL) | (df[c] > UL), c].count() if isnum else ''
        rtn.loc[c, 'noutlier(%)'] = np.round((df.loc[(df[c]<LL)|(df[c]>UL), c].count()/df[c].count())*100, round)  if isnum else ''
        rtn.loc[c, 'top'] = np.round(Q3+1.5*(Q3-Q1), round) if isnum else ''
        rtn.loc[c, 'bottom'] = np.round(Q1-1.5*(Q3-Q1), round) if isnum else ''
        rtn.loc[c, 'ntop'] = np.sum(df[c] > UL) if isnum else ''
        rtn.loc[c, 'nbottom'] = np.sum(df[c] < LL) if isnum else ''
    return rtn.sort_values('noutlier', ascending=False) if sort else rtn

import scipy.stats as stats
from sklearn.preprocessing import OrdinalEncoder
def eda_corr(df, target, round=3):
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
    return rtn

# 카테고리 변수 그룹별 평균의 차이 비교
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
def eda_anova(df, groups, endog):
    model = ols(endog+'~ C('+groups+')', df).fit()
    al = anova_lm(model)
    print(al)
    if al.iloc[0, -1] < 0.05:
        tmp = df[[endog, groups]].dropna()
        posthoc = pairwise_tukeyhsd(endog=tmp[endog], groups=tmp[groups], alpha=0.05)
        print('\n', posthoc)
    return al.iloc[0, -1]

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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

def plot_decision_boundaries(X, y, model_class, **model_params):
    X = np.array(X)
    y = np.array(y).flatten()
    reduced_data = X[:, :2]
    model = model_class(**model_params).fit(reduced_data, y)

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

from statsmodels.stats.outliers_influence import variance_inflation_factor
def reg_vif(df, sort=True):
    rtn = pd.DataFrame(
        data = [ variance_inflation_factor(df.values, i) for i in range(df.shape[1]) ],
        columns = ['VIF'],
        index = df.columns
        )
    return rtn.sort_values('VIF', ascending=False) if sort else rtn

def reg_equation(name, model, round=3):
    rtn = pd.DataFrame(
        data=np.round(model.coef_, round).reshape(1,-1),\
        columns=model.feature_names_in_,\
        index=[name])
    rtn['Intercept'] = np.round(model.intercept_, round)
    return rtn[np.insert(rtn.columns[:-1], 0, rtn.columns[-1])]

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
def reg_score(name, y_test, y_pred):
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

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import SGDRegressor, BaggingRegressor, RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor
def reg_model_test(X, y, train_size=.75, sort=True):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=13)
    models = {
        'LinearRegression' : LinearRegression(fit_intercept=True),
        'Lasso' : Lasso(alpha=0, max_iter=10000),
        'Ridge' : Ridge(alpha=1, max_iter=10000),
        'SGDRegressor' : SGDRegressor(penalty='l1', max_iter = 50, alpha = 0.001, early_stopping = True, n_iter_no_change = 3),
        'MLPRegressor' : MLPRegressor(max_iter = 5000, alpha = 0.1, verbose = False, early_stopping = True, hidden_layer_sizes = (100, 10)),
        'KNeighborsRegressor' : KNeighborsRegressor(n_neighbors = 5),
        'SVR' : SVR(),
        'DecisionTreeRegressor' : DecisionTreeRegressor(),
        'BaggingRegressor' : BaggingRegressor(),
        'RandomForestRegressor' : RandomForestRegressor(),
        'ExtraTreesRegressor' : ExtraTreesRegressor(),
        'AdaBoostRegressor' : AdaBoostRegressor(),
        'GradientBoostingRegressor' : GradientBoostingRegressor(),
    }
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        results.append(reg_score(name, y_test, model.predict(X_test)).T)
    return pd.concat(results).sort_values('r2', ascending=False) if sort else pd.concat(results)

def reg_poly_fit(X, y, degree, plot=True):
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

def reg_poly_resid(X, y, degree, scoring='se', plot=True):
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
        color = sns.color_palette('tab10')
        for i, c in enumerate(rtn.columns):
            plt.plot(rtn.index, rtn[c], color=color[i], alpha=.5, label=c)
        plt.title(f'Reg Error(Degree 1 ~ {degree})')
        plt.xlabel('Features')
        plt.ylabel('Absolute Errors')
        plt.legend()
        plt.show()
    return rtn

import pandas as pd
import numpy as np
import seaborn as sns
def feature_importance(df, model, sort=True, plot=True):
    rtn = pd.DataFrame(data=np.array(model.feature_importances_), index=df.columns, columns=['importances'])
    if sort:
        rtn = rtn.sort_values(by='importances', ascending=False)
    if plot:
        sns.barplot(y=rtn.index, x=rtn.importances, palette='Set2')
    return rtn