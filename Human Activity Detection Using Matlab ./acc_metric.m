function [accuracy,precision,recall,f1_score]=acc_metric(ytest,y_pred)
cm=confusionmat(ytest,y_pred)
cmt = cm'

diagonal = diag(cmt)
sum_of_rows = sum(cmt, 2)

precision = diagonal ./ sum_of_rows
overall_precision = mean(precision)

sum_of_columns = sum(cmt, 1)

recall = diagonal ./ sum_of_columns'
overall_recall = mean(recall)

f1_score = 2*((overall_precision*overall_recall)/(overall_precision+overall_recall))
accuracy=(sum(y_pred==ytest)/size(ytest,1))*100;
