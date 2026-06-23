from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

actual = [0, 0, 1, 1, 0, 1, 0, 1]
predicted = [0, 1, 1, 1, 0, 0, 0, 1]

cm = confusion_matrix(actual, predicted)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("2-Class Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.show()
