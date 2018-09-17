import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


color = ["red", "green", "blue", "magenta", "brown", "cyan"]
marker = ['o', 's', 'p', 'd', '>', '<']
weatset = ["Family", "Career", "Science", "Arts"] #, "Math", "Weapons"]


def compute_topic_assoc(gender_profile_lang, gender, gender_words, subject):
    
    topic_result = gender_profile_lang.data_store[gender]
    #print(topic_result)
    
    subject_bias = []
    for k in range(len(gender_words)):
        topic_dict = topic_result[gender_words[k]]
        topic_sim = topic_dict["WEAT_Topic_"+subject]

        #print(topic_sim)
        if len(topic_sim) > 0 and not np.isnan(np.sum(topic_sim)):
            subject_bias.append(topic_sim)

        #break #remove for full run
    subject_bias = np.array(subject_bias)
    avg_bias = np.average(subject_bias, axis=0)
    return avg_bias

    
def plot_male_female_association(gender_profile_lang):
    
    regression_params = []
    topic_association = []
    year = [1800+x for x in range(gender_profile_lang.start, gender_profile_lang.end, 10)]
    
    female_topic = gender_profile_lang.female_domain[0]
    female_words = gender_profile_lang.female_domain[1:]
    
    male_topic = gender_profile_lang.male_domain[0]
    male_words = gender_profile_lang.male_domain[1:]
    
    i = 0
    for subject in weatset: #dataset:
        plt.figure(figsize=(8, 6))
        
        female_topic_assoc = compute_topic_assoc(gender_profile_lang, female_topic, female_words, subject)
        male_topic_assoc = compute_topic_assoc(gender_profile_lang, male_topic, male_words, subject)
        
        slope, intercept, r_value, p_value, std_err = linregress(year, female_topic_assoc)
        regression_params.append([subject, female_topic, \
                                  slope, intercept, r_value, p_value, std_err])
        topic_association.append([subject, female_topic] + female_topic_assoc.tolist())
        
        #print(slope, intercept, r_value, p_value, std_err)
        regress_assoc = slope * np.array(year) + intercept
        
        plt.plot(regress_assoc, linestyle='-', linewidth=3, color='magenta')
        plt.plot(female_topic_assoc, color='magenta', marker='o', \
                linestyle='-', linewidth=4, alpha=0.25)
        plt.plot(female_topic_assoc, label="Female", marker='o', linewidth=0.1, \
             mfc='magenta', ms=6, mec='black', mew=1.25)
        
        slope, intercept, r_value, p_value, std_err = linregress(year, male_topic_assoc)
        regression_params.append([subject, male_topic, \
                                  slope, intercept, r_value, p_value, std_err])
        topic_association.append([subject, male_topic] + male_topic_assoc.tolist())
        
        #print(slope, intercept, r_value, p_value, std_err)
        regress_assoc = slope * np.array(year) + intercept
        
        plt.plot(regress_assoc, linestyle='-', linewidth=3, color='green')
        plt.plot(male_topic_assoc, color='green', marker='s', \
                linestyle='-', linewidth=4, alpha=0.25)
        plt.plot(male_topic_assoc, label="Male", marker='s', linewidth=0.1, \
             mfc='green', ms=6, mec='black', mew=1.25)
        
        i += 1
        
        plt.title("Gender Association for: " + subject)

        xaxis = [i for i in range(0, int((gender_profile_lang.end-gender_profile_lang.start)/10))]
        xtick = [1800+x for x in range(gender_profile_lang.start, gender_profile_lang.end, 10)]

        plt.xticks(xaxis, xtick, rotation=45)
        plt.legend(bbox_to_anchor=(0., 1.1, 1., .102), loc=4, \
                   ncol=2, mode="expand", borderaxespad=0.)
        #plt.legend(loc=2)
        #plt.ylim(-0.1, +0.1)
        plt.show()

    return regression_params, topic_association


def compute_bias_against_weat(gender_profile_lang):

    bias_scores = []    
    regression_params = []
    year = [1800+x for x in range(gender_profile_lang.start, gender_profile_lang.end, 10)]

    female_topic = gender_profile_lang.female_domain[0]
    female_words = gender_profile_lang.female_domain[1:]
    
    male_topic = gender_profile_lang.male_domain[0]
    male_words = gender_profile_lang.male_domain[1:]
    
    i = 0 
    plt.figure(figsize=(8, 6))
    for subject in weatset: #dataset:
        
        female_topic_assoc = compute_topic_assoc(gender_profile_lang, female_topic, female_words, subject)
        male_topic_assoc = compute_topic_assoc(gender_profile_lang, male_topic, male_words, subject)
           
        topic_bias = np.subtract(male_topic_assoc, female_topic_assoc)
       
        slope, intercept, r_value, p_value, std_err = linregress(year, topic_bias)
        regression_params.append([female_topic+"_"+male_topic, subject, \
                                  slope, intercept, r_value, p_value, std_err])
        bias_scores.append([female_topic+"_"+male_topic, subject] + topic_bias.tolist())
        
        #print(slope, intercept, r_value, p_value, std_err)
        regress_bias = slope * np.array(year) + intercept
        
        plt.plot(regress_bias, linestyle='-', linewidth=3, color=color[i])
        
        plt.plot(topic_bias, color=color[i], marker=marker[i], ms=6, \
                linestyle='-', linewidth=4, alpha=0.4)
        plt.plot(topic_bias, label=subject, marker=marker[i], linewidth=0.1, \
                 mfc=color[i], ms=6, mec='black', mew=1.25)
        i += 1
        """
        print("Analyzing Gender Bias for: ", subject)
        print("--------------------------------------")
        print("Gender Bias Scores: ", ",".join(str(x) for x in topic_bias))
        print("Regression Params: ", ",".join(str(x) for x in regression_params[-1]))
        print()
        """
        plt.title("Gender Bias (Male - Female): " + subject)
        #plt.title("Gender Bias for: " + lang_list[i])

        xaxis = [i for i in range(0, int((gender_profile_lang.end-gender_profile_lang.start)/10))]
        xtick = [1800+x for x in range(gender_profile_lang.start, gender_profile_lang.end, 10)]

        plt.xticks(xaxis, xtick, rotation=45)
        plt.ylim(-0.1, +0.1)
    
    plt.show()
        
    return regression_params, bias_scores


def compute_bias_without_plot(gender_profile_lang):

    bias_scores = []    
    regression_params = []
    year = [1800+x for x in range(gender_profile_lang.start, gender_profile_lang.end, 10)]

    female_topic = gender_profile_lang.female_domain[0]
    female_words = gender_profile_lang.female_domain[1:]
    
    male_topic = gender_profile_lang.male_domain[0]
    male_words = gender_profile_lang.male_domain[1:]
    
    i = 0 
    for subject in weatset[:1]: #dataset:
        
        female_topic_assoc = compute_topic_assoc(gender_profile_lang, female_topic, female_words, subject)
        male_topic_assoc = compute_topic_assoc(gender_profile_lang, male_topic, male_words, subject)
           
        topic_bias = np.subtract(male_topic_assoc, female_topic_assoc)
        assert(len(year) == len(topic_bias)) 
        slope, intercept, r_value, p_value, std_err = linregress(year, topic_bias)
        regression_params.append(["Male-Female", subject, \
                                  slope, intercept, r_value, p_value, std_err])
        bias_scores.append(["Male-Female", subject] + topic_bias.tolist())
        
    return regression_params, bias_scores



def evaluate_bias_subject_language(gender_profile_lang):
    
    topic_bias_year = []
    regression_params = []
    year = [1800+x for x in range(gender_profile_lang.start, gender_profile_lang.end, 10)]
   
    female_topic = gender_profile_lang.female_domain[0]
    female_words = gender_profile_lang.female_domain[1:]
    
    male_topic = gender_profile_lang.male_domain[0]
    male_words = gender_profile_lang.male_domain[1:]
    
    for subject in weatset: #dataset:
        female_topic_assoc = compute_topic_assoc(gender_profile_lang, female_topic, female_words, subject)
        male_topic_assoc = compute_topic_assoc(gender_profile_lang, male_topic, male_words, subject)
         
        topic_bias = np.subtract(male_topic_assoc, female_topic_assoc)
        topic_bias_year.append(topic_bias)
        
        slope, intercept, r_value, p_value, std_err = linregress(year, topic_bias)
        regression_params.append([female_topic+"_"+male_topic, subject, \
                                  slope, intercept, r_value, p_value, std_err])

    return (topic_bias_year, regression_params)


class ListTable(list):
    """ Overridden list class which takes a 2-dimensional list of 
        the form [[1,2,3],[4,5,6]], and renders an HTML Table in 
        IPython Notebook. """
    
    def __init__(self, shift):
        self.shift = shift
    
    def _repr_html_(self):
        html = ["<table>"]
        index = 0
        
        for row in self:
            #html.append("<tr>")
            
            if self.shift:
                if index%2 == 1:
                    html.append("<tr>")
                    html.append("<td rowspan=\"2\">{0}</td>".format(row[0]))
                else:
                    html.append("<tr>")
                    if index == 0:
                        html.append("<td>{0}</td>".format("subject"))

            index += 1 
            
            for col in row[self.shift:]:
                html.append("<td>{0}</td>".format(col))
            
            html.append("</tr>")
        html.append("</table>")
        return ''.join(html)

