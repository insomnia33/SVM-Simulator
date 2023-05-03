import streamlit                as st
import pandas                   as pd
import numpy                    as np
import plotly.express           as px
import plotly.graph_objects     as go

from sklearn                    import svm
from sklearn.metrics            import classification_report, mean_absolute_percentage_error

def LoadTipsData():
    dataframe = px.data.tips()
    st.dataframe(dataframe, height=200)
    return dataframe


def LoadIrisData():
    dataframe = px.data.iris()
    dataframe['species'] = dataframe.species.map({'setosa':1, 'versicolor':2,'virginica':3})

    st.dataframe(dataframe, height=200)
    return dataframe

def TipsSVR(df, params):
    X = df['total_bill'].values.reshape(-1, 1)
    y = df['tip']


    model = svm.SVR(kernel=params['kernel'], C=params['c'], gamma=params['gamma'])
    model.fit(X, y)

    tipsScore = mean_absolute_percentage_error(y, model.predict(X))

    tipsScore = f'MAPE: {tipsScore:.2f}'

    return X, y, tipsScore, model

def TipsPlot(df, X, y, model):

    x_range = np.linspace(X.min(), X.max(), 100)
    y_range = model.predict(x_range.reshape(-1, 1))

    fig = px.scatter(df, x='total_bill', y='tip', opacity=0.65)
    fig.add_traces(go.Scatter(x=x_range, y=y_range, name='Regression Fit'))

    return fig

def IrisSVC(df, params):
    X = df[params['cols']]
    y = df['species']

    model = svm.SVC(kernel=params['kernel'], C=params['c'], gamma=params['gamma'])
    model.fit(X, y)
    score = classification_report(y, model.predict(X), output_dict=True)

    return X, y, model, score


def IrisSVR(df, params):
    X = df[params['cols'][:-1]]
    y = df[params['cols'][-1]]

    model = svm.SVR(kernel=params['kernel'], C=params['c'], gamma=params['gamma'])
    model.fit(X, y)

    score = mean_absolute_percentage_error(y, model.predict(X))
    score = f'MAPE: {score:.2f}'

    return X, y, model, score



def IrisPlot2D(X, y, model, params):

    mesh_size = .02
    margin = 0

    # Create a mesh grid on which we will run our model

    x_min, x_max = X[params['cols'][0]].min() - margin, X[params['cols'][0]].max() + margin
    y_min, y_max = X[params['cols'][1]].min() - margin, X[params['cols'][1]].max() + margin
    xrange = np.arange(x_min, x_max, mesh_size)
    yrange = np.arange(y_min, y_max, mesh_size)
    xx, yy = np.meshgrid(xrange, yrange)

    # Run model
    pred = model.predict(np.c_[xx.ravel(), yy.ravel()])
    pred = pred.reshape(xx.shape)

    fig = px.scatter(X, x=params['cols'][0], y=params['cols'][1], color=y)
    
    fig.add_trace(go.Contour(x=xx[0], y=yy[:,0], z=pred, opacity=0.5, showscale=False, colorscale='Jet'))
    fig.update_layout(width=800, height=600, coloraxis_showscale=False)

    return fig

def IrisPlot3D(X, y, model, params):

    mesh_size = .02
    margin = 0

    # Create a mesh grid on which we will run our model
    x_min, x_max = X[params['cols'][0]].min() - margin, X[params['cols'][0]].max() + margin
    y_min, y_max = X[params['cols'][1]].min() - margin, X[params['cols'][1]].max() + margin
    xrange = np.arange(x_min, x_max, mesh_size)
    yrange = np.arange(y_min, y_max, mesh_size)
    xx, yy = np.meshgrid(xrange, yrange)

    # Run model
    pred = model.predict(np.c_[xx.ravel(), yy.ravel()])
    pred = pred.reshape(xx.shape)

    # Generate the plot
    fig = px.scatter_3d(X, x=params['cols'][0], y=params['cols'][1], z=y)
    fig.update_traces(marker=dict(size=5))
    fig.add_traces(go.Surface(x=xrange, y=yrange, z=pred, name='Regression Surface'))
    fig.update_layout(width=800, height=600)

    return fig


def LoadXORdata():
    data = [(0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 1)]
    dataframe = pd.DataFrame(data, columns=['feature1', 'feature2', 'class']) 
    return dataframe

def XORSVC(df, params):
    X = df[['feature1', 'feature2']]
    y = df['class']

    model = svm.SVC(kernel=params['kernel'], C=params['c'], gamma=params['gamma'])
    model.fit(X, y)

    return X, y, model

def XORPlot(df, X, y, model):

    mesh_size = .02
    margin = 0

    # Create a mesh grid on which we will run our model
    x_min, x_max = X.feature1.min() - margin, X.feature1.max() + margin
    y_min, y_max = X.feature2.min() - margin, X.feature2.max() + margin
    xrange = np.arange(x_min, x_max, mesh_size)
    yrange = np.arange(y_min, y_max, mesh_size)
    xx, yy = np.meshgrid(xrange, yrange)

    # Run model
    pred = model.predict(np.c_[xx.ravel(), yy.ravel()])
    pred = pred.reshape(xx.shape)


    # Generate the plot
    fig = px.scatter(df, x='feature1', y='feature2', color='class', title='XOR Problem')
    fig.update_traces(marker=dict(size=33))
    fig.add_trace(go.Contour(x=xx[0], y=yy[:,0], z=pred, opacity=0.5, showscale=False, colorscale='Viridis'))
    fig.update_layout(width=800, height=600, coloraxis_showscale=False)
    return fig
    
def LoadCanceCellData():
    dataframe = pd.read_csv("cell_samples.csv")
    st.dataframe(dataframe, height=200)
    return dataframe



datasetList = [
    'Tips',
    'Iris',
    'XOR',
    'CancerCell'
]


st.sidebar.title('Inputs')

dataOption = st.sidebar.selectbox('Select a Dataset', options=datasetList, index=1)

match dataOption:
    case 'Tips':
        tipsData = LoadTipsData()
        tipsCols = tipsData.columns.to_list()
        tipsCols = st.sidebar.multiselect('Tips Features', options=tipsCols, default=tipsCols[:2])
    
        cParam = st.sidebar.number_input('C', min_value=0.0, step=0.1, value=10.0)
        gammaParam = st.sidebar.number_input('Gamma', min_value=0.0, step=0.1, format='%.1f', value=0.1)
        kernelParam = st.sidebar.selectbox('Kernel', options=['linear', 'rbf'])
        

        trainModel = st.sidebar.button('Train SVR')
        params = {
            'cols' : tipsCols,
            'c' : cParam,
            'gamma' : gammaParam,
            'kernel' : kernelParam
        }

        if trainModel:
            X, y, tipsScore, tipsModel = TipsSVR(tipsData, params)
            st.text(tipsScore)
            tipsPlot = TipsPlot(tipsData, X, y, tipsModel)
            st.plotly_chart(tipsPlot, use_container_width=True, theme=None)


    case 'Iris':

        irisData = LoadIrisData()
        irisCols = irisData.drop(columns=['species', 'species_id']).columns.to_list()
        irisCols = st.sidebar.multiselect('Iris Features', options=irisCols, default=irisCols[:2])
    
    
        cParam = st.sidebar.number_input('C', min_value=0.0, step=0.1, value=10.0)
        gammaParam = st.sidebar.number_input('Gamma', min_value=0.0, step=0.1, format='%.1f', value=0.1)
        kernelParam = st.sidebar.selectbox('Kernel', options=['linear', 'rbf', 'poly'])
        
        col1, col2= st.sidebar.columns(2)

        trainModel = col2.button('Train')
        svmBut = col1.radio('',options=['SVC', 'SVR'], horizontal=True)

        params = {
            'type': svmBut,
            'cols' : irisCols,
            'c' : cParam,
            'gamma' : gammaParam,
            'kernel' : kernelParam
        }

        if trainModel:
            if svmBut == 'SVC':
                X, y, irisModel, irisScore = IrisSVC(irisData, params)
                st.table(irisScore)

                if len(params['cols']) == 2:
                    irisPlot = IrisPlot2D(X, y, irisModel, params)
                    st.plotly_chart(irisPlot)

                if len(params['cols']) == 3:
                    irisPlot = IrisPlot3D(X, y, irisModel, params)
                    st.plotly_chart(irisPlot)

            if svmBut == 'SVR':
                X, y, irisModel, irisScore = IrisSVR(irisData, params)
                st.text(irisScore)
    

                if len(params['cols']) == 3:
                    irisPlot = IrisPlot3D(X, y, irisModel, params)
                    st.plotly_chart(irisPlot)
           

        

    case 'XOR':
        xorData = LoadXORdata()
        st.dataframe(xorData)
        
        cParam = st.sidebar.number_input('C', min_value=0.0, step=0.1, value=10.0)
        gammaParam = st.sidebar.number_input('Gamma', min_value=0.0, step=0.1, format='%.1f', value=0.1)
        kernelParam = st.sidebar.selectbox('Kernel', options=['linear', 'rbf', 'poly'])
        
        trainModel = st.sidebar.button('Train SVC')

        params = {
            'c' : cParam,
            'gamma' : gammaParam,
            'kernel' : kernelParam
        }

        if trainModel:
            X, y, XORModel = XORSVC(xorData, params)
            xorScore = classification_report(y, XORModel.predict(X), output_dict=True)
            st.table(xorScore)
            XORPlot = XORPlot(xorData, X, y, XORModel)
            st.plotly_chart(XORPlot, use_container_width=True, theme=None)
                


    case 'CancerCell':
        cellData = LoadCanceCellData()






