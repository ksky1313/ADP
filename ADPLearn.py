# 기본
import numpy as np
import pandas as pd

# 정규화 
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 주성분 분석
from sklearn.decomposition import PCA

# 요인분석
import pingouin as pg
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo
from factor_analyzer import FactorAnalyzer
from sklearn.decomposition import FactorAnalysis

# 데이터셋
from sklearn.datasets import load_iris

# 시각화
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (7, 5)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False


def load_data(options='iris'):
    """ADP 학습, 테스트 데이터 제공

    Args:
        options (str, optional): (Default) iris 
            Data 옵션 : iris | mtcars | usarrests | swiss | titanic | baseball | cars93 | airquality

    Returns:
        train (DataFrame): data DataFrame
        test (DtaFrame): target DataFrame
    """
    train = pd.DataFrame()
    test = pd.DataFrame()

    data_url = {
        'usarrests':'https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/USArrests.csv',
        'swiss':'https://gist.githubusercontent.com/christophsax/178d34245afdd6e187b1fff72dbe7448/raw/f5f4189f949f117bee4e82e4aa75c104ed20b4f4/swiss.csv',
        'mtcars':'https://gist.githubusercontent.com/seankross/a412dfbd88b3db70b74b/raw/5f23f993cd87c283ce766e7ac6b329ee7cc2e1d1/mtcars.csv',
        'titanic':'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv',
        'baseball':'https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/plyr/baseball.csv',
        'cars93':'https://raw.githubusercontent.com/selva86/datasets/master/Cars93.csv',
        'airquality':'https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/airquality.csv',
    }

    if options in data_url:
        train=pd.read_csv(data_url[options])
    else:
        sns.load_dataset('iris')
        train = sns.load_dataset('iris')
        
    return train, test



def scaler(df, options='minmax'):
    """데이터 정규화

    Args:
        df (DataFrame): Original DataFrame
        options (str, optional): (Defaults)'minmaxs' | 'Standard'

    Returns:
        scaled_df (DataFrame): Transformed DataFrame
    """
    scaled_df = df.copy()

    if options == 'standard':
        sc = StandardScaler()
    elif options == 'minmax':
        sc = MinMaxScaler()
    else:
        print("Error >> Choose : standard : minmax")
        return None

    for c in scaled_df.columns[scaled_df.dtypes != 'object']:
        scaled_df[c] = sc.fit_transform(scaled_df[[c]])

    return scaled_df



def test_fa(df, options='bartlett'):
    """요인분석 적정성 검사

    Args:
        df (DataFrame): Test target DataFrame
        options (str, optional): (Default)'bartlett' | 'kmo'

    Returns:
        t_r (DataFrame): Transformed DataFrame
        t_pass (bool): Test result
    """

    t_r, t_pass = [], False

    # 관측된 변수들의 상호 연관성 확인
    if options == 'bartlett':
        chi_square_value, p_value = calculate_bartlett_sphericity(df)
        t_r.append(['chi_square_value', chi_square_value])
        t_pass = (p_value < 0.05 )
    # 과늑 변수와 전체 모형의 적합성 결정(0~1 사이, 0.6미만이면 부적합)
    elif options == 'kmo':
        kmo_all, kmo_model=calculate_kmo(df)
        t_r.append(['kmo_all', kmo_all])
        t_pass = (kmo_model >= 0.6 )

    return t_r, t_pass


def factor_analysis(df, cev:float=0.85):
    """요인 분석

    Args:
        df (DataFrame): Original DataFrame
        cev (Float, optional) : (Default) 0.85, Minimum Cumulative Eigen Value

    Returns:
        ncom (Int): the number of Factor Analysis components
        cev (Float): Cumulative Eigen Value
        lodings (DataFrame): components_.T result
        fa_df (DataFrame): Transformed DataFrame

    """
    non_category = df.columns[df.dtypes != 'object']
    tdf = pd.DataFrame(data=df[non_category], columns = non_category)

    # 최대 갯수로 요인분석 실행
    ncomp = tdf.shape[1]
    fa = FactorAnalyzer()
    fa.fit(tdf, ncomp)
    ev, v = fa.get_eigenvalues()

    # 차트로 표현
    x = range(1, ncomp+1)
    y = ev
    s = np.cumsum(ev)
    e = [cev*ncomp] * ncomp

    plt.title('Factor Analysis [ ncomp {} ]'.format(ncomp))
    plt.ylim(0,ncomp+1)
    plt.xticks(x)
    plt.xlabel('Components')
    plt.ylabel('Cumulative Eigen Value')
    plt.grid(axis='y', color='blue', alpha=0.2, linestyle=':')

    plt.plot(x, e, 'g:', alpha=0.5)
    plt.text(x[0]-0.01, e[0], '{:.2f}'.format(cev), ha='right', color='g')

    plt.bar(x, y, color='grey', width=0.4, alpha=0.2)
    plt.plot(x, s, 'bo-')

    n_components = -1
    for idx, cs in enumerate(s):
        plt.text(x[idx], 0.3, 'fa{}'.format(idx+1), ha='center')

        if cev*ncomp <= cs and n_components == -1:
            plt.text(x[idx], cs+0.2, '{:.2f}'.format(cs), ha='center', color='r')
            n_components = idx
        else:
            plt.text(x[idx], cs+0.2, '{:.2f}'.format(cs), ha='center', color='b')

    plt.show()
    
    # 최적 갯수 요인 분석 실행
    fa = FactorAnalysis(n_components=n_components+1)
    fa_comp = fa.fit_transform(tdf)

    # 결과 리턴
    columns = columns=[ 'fa_' + str(i) for i in range(1, n_components+2) ]
    loadings = pd.DataFrame(data=fa.components_.T, index=tdf.columns, columns=columns)
    fa_df = pd.DataFrame(data=fa_comp, index=tdf.index, columns=columns)
    return n_components+1, s[n_components], loadings, fa_df


def cronbach_alpha(df, loadings, min_loadings:float=0.7):
    """크론바흐 알파 : 요인분석 결과 검정에 활용

    Args:
        df (_type_): Original DataFrame
        loadings (_type_): The loading value DataFrame as a result of a factor analysis
        min_loadings (float, optional): (Defaults) 0.7, Minimum loading value

    Returns:
        t_r (list): 요인별 크론바흐 알파 값
    """

    t_r = []

    for c in loadings.columns:
        idx = loadings[c].loc[loadings[c] > min_loadings].index

        # 적어도 변수가 2개 이상이어야 크론바흐 알파 값 계산 가능
        if len(idx) >= 2:
            factor_alpha = pg.cronbach_alpha(df[idx])
            t_r.append({c:factor_alpha})

    return t_r




def pca(df, cev:float=0.9):
    """주성분 분석

    Args:
        df (DataFrame): Original DataFrame
        cev (Float, optional) : (Default) 0.9, Minimum cumulative explained value

    Returns:
        ncom (Int): the number of PCA components
        cev (Float): Cumulative Explained Value
        lodings (DataFrame): components_.T result
        pca_df (DataFrame): Transformed DataFrame
    """
    non_category = df.columns[df.dtypes != 'object']
    tdf = pd.DataFrame(data=df[non_category], columns = non_category)

    # 최대 갯수로 PCA 실행
    ncomp = min(tdf.shape[0], tdf.shape[1])
    pca = PCA(n_components = ncomp)
    pca.fit(tdf)

    # 차트로 표현
    x = range(1, ncomp+1)
    y = pca.explained_variance_ratio_ 
    s = np.cumsum(y)
    e = [cev] * ncomp

    plt.title('PCA [ ncomp {} ]'.format(ncomp))
    plt.ylim(0,1.2)
    plt.xticks(x)
    plt.xlabel('Components')
    plt.ylabel('Cumulative Explained Value')
    plt.grid(axis='y', color='blue', alpha=0.2, linestyle=':')
    
    plt.plot(x, e, 'g:', alpha=0.5)
    plt.text(x[0]-0.01, cev, '{:.2f}'.format(cev), ha='right', color='g')
    
    plt.bar(x, y, color='grey', width=0.4, alpha=0.2)
    plt.plot(x, s, 'bo-')

    n_components = -1
    for idx, cs in enumerate(s):
        plt.text(x[idx], 0.03, 'pca{}'.format(idx+1), ha='center')

        if cev <= cs and n_components == -1:
            plt.text(x[idx], cs+0.04, '{:.4f}'.format(cs), ha='center', color='r')
            n_components = idx
        else:
            plt.text(x[idx], cs+0.04, '{:.4f}'.format(cs), ha='center', color='b')

    plt.show()

    # 최적 갯수 주성분 분석 실행
    pca = PCA(n_components=n_components+1)
    pca_comp = pca.fit_transform(tdf)

    # 결과 리턴
    columns = columns=[ 'pca_' + str(i) for i in range(1, n_components+2) ]
    loadings = pd.DataFrame(data=pca.components_.T, index=tdf.columns, columns=columns)
    pca_df = pd.DataFrame(data=pca_comp, index=tdf.index, columns=columns)
    return n_components+1, s[n_components], loadings, pca_df