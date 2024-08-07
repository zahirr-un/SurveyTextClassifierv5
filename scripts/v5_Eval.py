from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Assuming y_test and y_test_pred are the actual and predicted labels for the test set
accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred, average='micro')
recall = recall_score(y_test, y_test_pred, average='micro')
f1 = f1_score(y_test, y_test_pred, average='micro')

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
