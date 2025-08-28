from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score,accuracy_score

# Clase de Árbol de decisión
class Decision_Tree:
    def __init__(self):
        self.model = DecisionTreeClassifier(max_depth=4,criterion='gini')
    def train_model(self,x_train,y_train):
        # Entrenamiento del modelo
        self.model.fit(x_train,y_train)
    def predict_data(self,x_test):
        # Predición
        y_test = self.model.predict(x_test)
        return y_test
    def get_probability(self,x_test):
        # Predecir probabilidades de tener 'Patología X'
        return self.model.predict_proba(x_test)
    def get_precision_score(self,x_test,y_test):
        # Valor de precisíon de la predicción
        y_pred = self.model.predict(x_test)
        return precision_score(y_test,y_pred)
    def get_acc_score(self,x_test,y_test):
        # Predicciones correctas
        y_pred = self.model.predict(x_test)
        return accuracy_score(y_test,y_pred)
        
    
    
