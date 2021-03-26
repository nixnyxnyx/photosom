import numpy as np
import altair as alt
import pandas as pd
from minisom import MiniSom
import scipy.stats as sps
from sklearn.ensemble import RandomForestClassifier
alt.data_transformers.disable_max_rows()

class ClassCond(object):
    def __init__(self, classifier, bins):
        """This is the random forest algorithm used for the machine learning section."""
        self.bins = bins
        self.clf = classifier

        
    def fit(self, X, Y, z_offset=False, Z=0): # X is photometry(array), Y is redshift, Z is for making sure dimensionality works
        if z_offset: z_total = Z # If the z_offset flag is TRUE, then I use 'Z' to substitute for Y
        else: z_total = Y # This basically means I use the redshift min/max range of Z instead of Y
        
        grid_Y = np.linspace(np.min(z_total), np.max(z_total), num=self.bins)
        self.delta_z = grid_Y[1:] - grid_Y[:-1]
        grid_Y = np.linspace(np.min(z_total)-self.delta_z[0]/4., np.max(z_total)+self.delta_z[0]/4., num=self.bins)
        
        self.midpoints = grid_Y[:-1] + self.delta_z/2
        self.response_classes = np.array(np.digitize(Y, bins=grid_Y), dtype=np.int)
        self.missing_no = np.setdiff1d(np.arange(1,100), self.response_classes)
        #put edge cases in bins in range

        self.clf.fit(X=X, y=self.response_classes)

        
    def predict(self, X): #dim(x) = nsamples, nbands
        nbatch = 10
        prob_vec = self.clf.predict_proba(X)
        #normalize the histogram to unit area (p * delta z)
        #prob vec is a matrix with each row being a galaxy and each column being a histogram bin height.
        result = prob_vec/(self.delta_z[0]) #on the premise that the grid is evenly spaced
        return result
    
    
    def ind_result(self,X):
        prob_vec = self.predict(X)
        each = sum(self.midpoints * prob_vec *(z_max/(n_bins-1)))

        
class PhotoSOM(object):
    def __init__(self, train, test, random_seed=None):
        """
        An object that takes in training and test datasets of galaxes for use in machine learning.
        Parameters:
        train: numpy array
            An array of length (number of phomometric bands + 1) which holds the redshift truth values
            in the first column and the photometric data in successive columns, each devoted to one band.
            Used for creating the ML model.
        test: numpy array
            Same dimensions as train, used for prediction.
        """
        
        # Since there are some 99.0 values in u-band, we need to exclude that so the SOM doesn't freak out
        self.train = train[train[:, 1] < 50.]
        self.test = test[test[:, 1] < 50.]
        
        self.train_z = self.train[:, 0] 
        self.test_z = self.test[:, 0]
        
        # u g r i z y by default
        self.train_phot = self.train[:, 1:]
        self.test_phot = self.test[:, 1:]
        
        self.selection_bins = 30
        self.ml_bins = 100
        self.quantile_cut = 0.0
        self.z_range = [np.min(np.append(self.train_z, self.test_z)), np.max(np.append(self.train_z, self.test_z))]
        self.mag_range = [np.min(np.append(self.train_phot, self.test_phot)), np.max(np.append(self.train_phot, self.test_phot))]
        
        self.random_seed = random_seed
        if random_seed != None:
            self.random_state = np.random.RandomState(random_seed)
        
        self.filter1 = 2
        self.filter2 = 3
        self.filter3 = 3
        
        
    def selFuncHelper(self, filter1, filter2):
        color = self.train[:, filter1]-self.train[:, filter2]
        redshift = self.train[:, 0]
        cut, edges = pd.qcut(color, q=self.selection_bins, retbins=True)
    
        result = self.train.copy()
        for i in range(len(edges)-1):
            bin_population = redshift[(color>edges[i])&(color<edges[i+1])]
            cutoff = np.quantile(bin_population, self.quantile_cut)
            mask = ~((color>edges[i])&(color<edges[i+1])&(redshift<cutoff))
            result = result[mask]
            color = color[mask]
            redshift = redshift[mask]
        return result
    
    
    def rangeFuncHelper(self, filter3):
        z_data = self.train[:, 0]
        mag_data = self.train[:, 1:]
        mask_z = (z_data > self.z_range[0]) & (z_data < self.z_range[1])
        mag_data_masked = mag_data[mask_z]
        mask_mag = (mag_data_masked[:,filter3] > self.mag_range[0]) & (mag_data_masked[:,filter3] < self.mag_range[1])
        return (z_data[mask_z][mask_mag], mag_data_masked[mask_mag])
    
    
    def selectionFunction(self, bins, quantile, filter1=2, filter2=3):
        self.selection_bins = bins
        self.quantile_cut = quantile
        self.filter1 = filter1
        self.filter2 = filter2
        self.transformData()

        
    def assignRange(self, z_range, mag_range, filter3=3):
        self.z_range = z_range
        self.mag_range = mag_range
        self.filter3 = filter3
        self.transformData()
    

    def transformData(self):
        sel_train = self.selFuncHelper(self.filter1, self.filter2)
        self.train_z, self.train_phot = self.rangeFuncHelper(self.filter3)
        
        # The number of galaxies represented in the train/test sets
        print("There are %d training galaxies selected"%self.train_z.shape[0])
    
    
    def idealGaussian(self, sample_sigma=0.02, pdf_sigma=0.2):
        gauss_ideal = np.empty(len(self.test_z), dtype='O')
        for i in range(len(gauss_ideal)):
            gauss_ideal[i] = sps.norm(loc=self.test_z[i], scale=sample_sigma)

        pdfs_ideal = np.empty(len(self.test_z), dtype='O')
        for i in range(len(gauss_ideal)):
            pdfs_ideal[i] = sps.norm(loc=gauss_ideal[i].rvs(random_state=random_state), scale=pdf_sigma)
        self.pdfs = pdfs_ideal
        return pdfs_ideal


    def randomForestTraining(self):
        # Create the model object and fit the training data
        self.model = ClassCond(RandomForestClassifier(n_jobs=-1), bins=self.ml_bins)
        self.model.fit(self.train_phot, self.train_z, z_offset=True, Z=self.train[:, 0])

        # Create predictions on test data and get useable statistics from it
        pred = self.model.predict(self.test_phot)

        if self.model.missing_no.shape[0]:
            print(str(self.model.missing_no.shape[0])+" bins missing, "+str(self.model.missing_no))
            for c in self.model.missing_no:
                pred = np.insert(pred, c-1, 0., axis=1)
        else:
            print("No bins missing")

        endpoints = np.append((self.model.midpoints - (self.model.midpoints[1]-self.model.midpoints[0])/2), \
                          self.model.midpoints[98] + (self.model.midpoints[1]-self.model.midpoints[0])/2)
        endpoints[0] = 0

        pred_hist = np.empty(len(pred), dtype='O')
        for i in range(len(pred_hist)):
            pred_hist[i] = sps.rv_histogram((pred[i], endpoints))
        self.pdfs = pred_hist
        return pred_hist
    
    
    def predictionPlot(self, bins=100, title="", pixel=300):
        pred_mean = []
        for i in range(len(self.pdfs)):
            pred_means[i] = self.pdfs[i].mean()
        
        z_range = [np.min(self.train[:,0]), np.max(self.train[:,0])]
        pz_data = pd.DataFrame({'true_z': self.test_z, 'predicted': pred_mean})
        unity = pd.DataFrame({'hor': z_range, 'ver': z_range})

        pz_plot = alt.Chart(pz_data).mark_rect().encode(
            alt.X('true_z:Q', bin=alt.Bin(extent=z_range, step=(z_range[1]-z_range[0])/bins),
                  axis=alt.Axis(title='True Redshift (z)')),
            alt.Y('predicted:Q', bin=alt.Bin(extent=z_range, step=(z_range[1]-z_range[0])/bins),
                 axis=alt.Axis(title='Predicted Redshift (z)')),
            alt.Color('count(true_z):Q', scale=alt.Scale(scheme='bluepurple', type='log'))
        ).properties(width=pixel, height=pixel, title=title)

        unity_plot = alt.Chart(unity).mark_line(color="#FF0000", strokeDash=[7,5], opacity=0.4).encode(x='hor', y='ver')

        return pz_plot + unity_plot
    
    
    def initSOM(self, size):
        self.size = size
        # Data assignment and normalization
        data = np.apply_along_axis(lambda x: x/np.linalg.norm(x), 1, self.test_phot)
        if self.random_seed != None:
            self.som = MiniSom(size, size, len(self.test_phot), sigma=1.0, learning_rate=0.5, random_seed=self.random_seed)
        else:
            self.som = MiniSom(size, size, len(self.test_phot), sigma=1.0, learning_rate=0.5)
        self.som.random_weights_init(data)
        self.som.train(data, data.shape[0])
        self.som_map = self.som.win_map(data)
        
        self.index_map = np.empty((size, size), dtype='O')
        for c in self.som_map:
            index = []
            for vec in self.som_map[c]:
                index.append(int(np.array(np.where(np.sum(data, axis=1) == np.sum(vec)))))
            self.index_map[c[1],c[0]] = index

    
    def colorMap(self, title="", pixel=300, scheme='purples', filter4=2, filter5=3):
        ri_colormap = np.zeros((self.size, self.size))
        for i in range(self.size):
            for j in range(self.size):
                if self.index_map[i,j]:
                    data_cut = self.test_phot[self.index_map[i,j]]
                    ri_color = data_cut[:,filter4] - data_cut[:,filter5]
                    ri_colormap[i,j] = np.mean(ri_color)
                    
        x, y = np.meshgrid(range(0, self.size), range(self.size-1,-1,-1))
        source = pd.DataFrame({'x': x.ravel(), 'y': y.ravel(), 'Color': ri_colormap.ravel()})
        return alt.Chart(source).mark_rect().encode(
            x=alt.X('x:O', axis=alt.Axis(labels=False, title=None)),
            y=alt.Y('y:O', axis=alt.Axis(labels=False, title=None)),
            color=alt.Color("Color:Q", scale=alt.Scale(scheme=scheme, type='linear'))
        ).properties(width=pixel,height=pixel,title=title)
    
    
    def densityMap(self, title="", pixel=300, scheme='inferno'):
        densitymap = np.zeros((self.size, self.size))
        for i in range(self.size):
            for j in range(self.size):
                densitymap[i,j] = len(self.index_map[i,j])
                
        x, y = np.meshgrid(range(0, self.size), range(self.size-1,-1,-1))
        source = pd.DataFrame({'x': x.ravel(), 'y': y.ravel(), 'Galaxies': densitymap.ravel()})
        return alt.Chart(source).mark_rect().encode(
            x=alt.X('x:O', axis=alt.Axis(labels=False, title=None)),
            y=alt.Y('y:O', axis=alt.Axis(labels=False, title=None)),
            color=alt.Color("Galaxies:Q", scale=alt.Scale(scheme=scheme, type='linear'))
        ).properties(width=pixel,height=pixel,title=title)
    
    
    def chiSquaredTest(self, bins=10, title="", pixel=300, scheme='greys'):
        null_map = np.zeros((self.size, self.size))
        for i in range(self.size):
            for j in range(self.size):
                cdf = []
                if self.index_map[i,j]:
                    pdf_cut = self.pdfs[self.index_map[i,j]]
                    truths = self.test_z[self.index_map[i,j]]
                    for k in range(len(pdf_cut)):
                        cdf.append(pdf_cut[k].cdf(truths[k]))
                        
                hist, endpoints = np.histogram(cdf, bins)
                chisq, p = sps.chisquare(hist)
                if p < 0.05:
                    null_map[i,j] = 1
        
        print("Failure Rate: " + str(100*np.sum(null_map)/(self.size**2)) + "%")
        x, y = np.meshgrid(range(0, self.size), range(self.size-1,-1,-1))
        source = pd.DataFrame({'x': x.ravel(), 'y': y.ravel(), 'Failed': null_map.ravel()})
        return alt.Chart(source).mark_rect().encode(
            x=alt.X('x:O', axis=alt.Axis(labels=False, title=None)),
            y=alt.Y('y:O', axis=alt.Axis(labels=False, title=None)),
            color=alt.Color("Failed:Q", scale=alt.Scale(scheme=scheme, type='linear'))
        ).properties(width=pixel,height=pixel,title=title)
    

    def chiSquaredMap(self, bins=10, title="", pixel=300, scheme='inferno'):
        chi_dof_map = np.zeros((size, size))
        for i in range(self.size):
            for j in range(self.size):
                cdf = []
                if self.index_map[i,j]:
                    pdf_cut = self.pdfs[self.index_map[i,j]]
                    truths = self.test_z[self.index_map[i,j]]
                    for k in range(len(pdf_cut)):
                        cdf.append(pdf_cut[k].cdf(truths[k]))
                        
                hist, endpoints = np.histogram(cdf, bins)
                chi_squared = 0
                for k in range(n_bins):
                    chi_squared += ((hist[k] - len(cdf) / bins)**2)/(len(cdf) / bins)
                chi_dof_map[i,j] = chi_squared/(bins-1)
        
        x, y = np.meshgrid(range(0, self.size), range(self.size-1,-1,-1))
        source = pd.DataFrame({'x': x.ravel(), 'y': y.ravel(), 'chisquared-dof': chi_dof_map.ravel()})
        return alt.Chart(source).mark_rect().encode(
            x=alt.X('x:O', axis=alt.Axis(labels=False, title=None)),
            y=alt.Y('y:O', axis=alt.Axis(labels=False, title=None)),
            color=alt.Color("chisquared-dof:Q", scale=alt.Scale(scheme=scheme, type='linear'))
        ).properties(width=pixel,height=pixel,title=title)
    
    
    def cdfHistogram(self, bins=30):
        cdfs = self.pdfs.cdf(self.test_z)
        return alt.Chart(pd.DataFrame({"CDF": cdfs})).mark_bar().encode(alt.X("CDF:Q", bin=alt.Bin(step=1./bins)), y='count()')