"""Seven anomaly-scoring approaches. Clustering scores are engineering proxies,
not supervised bot probabilities. Calibrate thresholds on a held-out unlabeled split.
"""
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering, Birch, DBSCAN, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.features import FEATURES

ALGORITHMS = ('isolation_forest', 'lof', 'kmeans', 'agglomerative', 'birch', 'dbscan')
# Six families PLUS Isolation Forest = six algorithms in total (LOF, KMeans,
# Agglomerative, BIRCH, DBSCAN, Isolation Forest).

class AnomalyModel:
    def __init__(self, algorithm='isolation_forest', contamination=.08, random_state=42):
        if algorithm not in ALGORITHMS:
            raise ValueError(f'Unknown algorithm {algorithm}; choose from {ALGORITHMS}')
        self.algorithm=algorithm
        self.contamination=contamination
        self.random_state=random_state

    def fit(self, X):
        self.preprocess_=ColumnTransformer([
            ('numeric',StandardScaler(),FEATURES[:-1]),
            ('categorical',OneHotEncoder(handle_unknown='ignore',sparse_output=False),['ua_family'])])
        z=self.preprocess_.fit_transform(X)
        self.fit_rows_=len(z)
        a=self.algorithm
        if a=='isolation_forest':
            self.estimator_=IsolationForest(n_estimators=150,contamination=self.contamination,random_state=self.random_state).fit(z)
        elif a=='lof':
            self.estimator_=LocalOutlierFactor(n_neighbors=min(25,len(z)-1),novelty=True,contamination=self.contamination).fit(z)
        elif a=='kmeans':
            self.estimator_=KMeans(n_clusters=8,random_state=self.random_state,n_init=10).fit(z)
        elif a=='agglomerative':
            # AgglomerativeClustering does not implement predict. Use training-cluster
            # centroids as a fixed out-of-sample distance proxy.
            labels=AgglomerativeClustering(n_clusters=8).fit_predict(z)
            self.centers_=np.vstack([z[labels==k].mean(axis=0) for k in np.unique(labels)])
        elif a=='birch':
            self.estimator_=Birch(n_clusters=8).fit(z)
            labels=self.estimator_.predict(z)
            self.centers_=np.vstack([z[labels==k].mean(axis=0) for k in np.unique(labels)])
        else:
            # DBSCAN is transductive. Nearest core sample distance gives a
            # fixed novelty proxy; if no cores exist, fall back to neighbor distance.
            self.estimator_=DBSCAN(eps=1.5,min_samples=10).fit(z)
            cores=self.estimator_.components_
            if len(cores)==0:
                self.no_cores_=True
                cores=z
            else:
                self.no_cores_=False
            self.neighbors_=NearestNeighbors(n_neighbors=1).fit(cores)
        return self

    def score_samples(self,X):
        """Higher = more anomalous; scores are NOT comparable across algorithms."""
        z=self.preprocess_.transform(X)
        a=self.algorithm
        if a in ('isolation_forest','lof'):
            return -np.asarray(self.estimator_.score_samples(z))
        if a=='kmeans':
            return np.min(self.estimator_.transform(z),axis=1)
        if a in ('agglomerative','birch'):
            return cdist(z,self.centers_).min(axis=1)
        return self.neighbors_.kneighbors(z,return_distance=True)[0].ravel()

    def calibrate(self, X):
        scores=self.score_samples(X)
        self.threshold_=float(np.quantile(scores,1-self.contamination))
        return self

    def predict(self,X):
        if not hasattr(self,'threshold_'):
            raise RuntimeError('Calibrate the model before predicting')
        return np.where(self.score_samples(X)>self.threshold_,-1,1)

    def decision_function(self,X):
        """Positive indicates normal, for backwards compatibility with starter API."""
        return self.threshold_-self.score_samples(X)

def build_pipeline(algorithm='isolation_forest'):
    return AnomalyModel(algorithm=algorithm)
