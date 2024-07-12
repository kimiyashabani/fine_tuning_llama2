from flask import Flask, request, jsonify
from gradientai import Gradient
import os


os.environ['GRADIENT_ACCESS_TOKEN'] = "mBleDSz9qO8uOVJurMYqyzZBNiIGXSjT"
os.environ['GRADIENT_WORKSPACE_ID'] = "92c98d43-5369-49c7-b625-27242f87df6e_workspace"

app = Flask(__name__)
new_model_adapter = None

def fine_tune_model():
    global new_model_adapter
    # LOAD AND SET THE MODEL
    with Gradient() as gradient:
        # SETTING UP THE BASE MODEL THAT WE WANT TO THE FINE-TUNING UN TOP OF IT
        # Nouse-hermes2 = FINE-TUNED VERSION OF LLAMA 2
        base_model = gradient.get_base_model(base_model_slug="nous-hermes2")
        new_model_adapter = base_model.create_model_adapter(name="test model 3")
        print(f"Created model adapter with id {new_model_adapter.id}")
        sample_query = "### Instruction: What is CHF short for? \n\n### Response:"
        print(f"Asking: {sample_query}")

        # BEFORE FINE-TUNING
        completion = new_model_adapter.complete(query=sample_query, max_generated_token_count=100).generated_output
        print(f"Generated (before fine-tune): {completion}")
        samples = [
            {
                "inputs": "### Instruction: What is hypertension? \n\n### Response: Hypertension, also known as high blood pressure, is a condition in which the force of the blood against the artery walls is too high."},
            {
                "inputs": "### Instruction: What are the symptoms of a heart attack? \n\n### Response: Symptoms of a heart attack can include chest pain, shortness of breath, nausea, and lightheadedness."},
            {
                "inputs": "### Instruction: How is atrial fibrillation treated? \n\n### Response: Atrial fibrillation can be treated with medications, electrical cardioversion, or ablation procedures."},
            {
                "inputs": "### Instruction: What is the normal range for blood pressure? \n\n### Response: The normal range for blood pressure is typically around 120/80 mmHg."},
            {
                "inputs": "### Instruction: Can you explain what an echocardiogram is? \n\n### Response: An echocardiogram is an ultrasound test that uses sound waves to create images of the heart."},
            {
                "inputs": "### Instruction: What is diabetes? \n\n### Response: Diabetes is a chronic condition that affects how your body turns food into energy, characterized by high blood sugar levels."},
            {
                "inputs": "### Instruction: How can one manage high cholesterol? \n\n### Response: High cholesterol can be managed through diet, exercise, and medications such as statins."},
            {
                "inputs": "### Instruction: What are the common signs of a stroke? \n\n### Response: Common signs of a stroke include sudden numbness or weakness in the face, arm, or leg, especially on one side of the body, confusion, trouble speaking, and difficulty walking."},
            {
                "inputs": "### Instruction: How is type 2 diabetes different from type 1 diabetes? \n\n### Response: Type 2 diabetes is often related to lifestyle factors and is characterized by insulin resistance, whereas type 1 diabetes is an autoimmune condition where the body does not produce insulin."},

            {
                "inputs": "### Instruction: What is hypertension? \n\n### Response: Hypertension, or high blood pressure, is a condition where the force of the blood against the artery walls is too high. Causes: Unhealthy diet, lack of physical activity, obesity, smoking, excessive alcohol consumption, stress, genetic factors. Treatment: Medications (ACE inhibitors, beta-blockers, diuretics), lifestyle changes. Prevention: Healthy diet, regular exercise, weight management, smoking cessation, stress management."
            },
            {
                "inputs": "### Instruction: Explain high blood pressure. \n\n### Response: High blood pressure, or hypertension, is a condition where the blood pressure in the arteries is persistently elevated. Causes: Unhealthy diet, lack of physical activity, obesity, smoking, excessive alcohol consumption, stress, genetic factors. Treatment: Medications (ACE inhibitors, beta-blockers, diuretics), lifestyle changes. Prevention: Healthy diet, regular exercise, weight management, smoking cessation, stress management."
            },
            {
                "inputs": "### Instruction: What is hypotension? \n\n### Response: Hypotension is abnormally low blood pressure, which can cause dizziness and fainting. Causes: Dehydration, heart problems, endocrine issues, severe infection (septicemia), blood loss. Treatment: Addressing the underlying cause, increasing salt intake, drinking more water, medications. Prevention: Staying hydrated, avoiding alcohol, wearing compression stockings."
            },
            {
                "inputs": "### Instruction: Define low blood pressure. \n\n### Response: Low blood pressure, or hypotension, is when blood pressure is lower than normal, which can lead to dizziness and fainting. Causes: Dehydration, heart problems, endocrine issues, severe infection (septicemia), blood loss. Treatment: Addressing the underlying cause, increasing salt intake, drinking more water, medications. Prevention: Staying hydrated, avoiding alcohol, wearing compression stockings."
            },
            {
                "inputs": "### Instruction: Explain cardiomegaly. \n\n### Response: Cardiomegaly refers to an enlarged heart, which can result from various conditions, including high blood pressure and heart valve disease. Causes: High blood pressure, heart valve disease, cardiomyopathy, pulmonary hypertension. Treatment: Medications (diuretics, ACE inhibitors, beta-blockers), treating the underlying cause, lifestyle changes."
            },
            {
                "inputs": "### Instruction: What is endocarditis? \n\n### Response: Endocarditis is an infection of the inner lining of the heart chambers and valves. Causes: Bacterial infection, fungal infection. Treatment: Antibiotics, antifungal medications, surgery (in severe cases). Prevention: Good dental hygiene, avoiding intravenous drug use, prompt treatment of infections."
            },
            {
                "inputs": "### Instruction: Define pericarditis. \n\n### Response: Pericarditis is inflammation of the pericardium, the thin sac surrounding the heart. Causes: Viral infections, bacterial infections, autoimmune diseases, heart attack, severe chest pain. Treatment: Anti-inflammatory medications, pain relievers, antibiotics (if bacterial), surgery (in severe cases). Prevention: Prompt treatment of infections, managing underlying conditions."
            },
            {
                "inputs": "### Instruction: What is myocarditis? \n\n### Response: Myocarditis is inflammation of the heart muscle, often caused by viral infections and resulting in symptoms that are similar to a heart attack. Causes: Viral infections, bacterial infections, autoimmune diseases, drug reactions. Treatment: Medications (anti-inflammatory drugs, antibiotics if bacterial), treating underlying conditions, rest. Prevention: Prompt treatment of infections, avoiding illicit drug use, vaccinations."
            },
            {
                "inputs": "### Instruction: Explain valvular heart disease. \n\n### Response: Valvular heart disease involves damage to or a defect in one of the four heart valves. Causes: Congenital defects, rheumatic fever, infections (endocarditis), aging. Treatment: Medications (diuretics, beta-blockers, ACE inhibitors), valve repair or replacement surgery."
            },
            {
                "inputs": "### Instruction: What is mitral valve prolapse? \n\n### Response: Mitral valve prolapse is a condition where the mitral valve doesn't close properly, sometimes causing blood to leak backward into the left atrium. Causes: Genetic factors, connective tissue disorders. Treatment: Monitoring, medications (beta-blockers), surgery (in severe cases). Prevention: Regular medical checkups, managing risk factors."
            },
            {
                "inputs": "### Instruction: What are the mitral valve prolapse symptoms? \n\n### Response: if symptoms occur, they include: rapid heartbeat, chest discomfort, and fatigue."
            },
            {
                "inputs": "### Instruction: Define aortic stenosis. \n\n### Response: Aortic stenosis is the narrowing of the aortic valve opening, restricting blood flow from the heart to the rest of the body. Causes: Congenital heart defect, calcium buildup on the valve, rheumatic fever. Treatment: Medications, valve repair or replacement surgery."
            },
            {
                "inputs": "### Instruction: Explain aortic aneurysm. \n\n### Response: An aortic aneurysm is an abnormal bulge in the wall of the aorta, which can rupture and cause life-threatening bleeding. Causes: Atherosclerosis, genetic conditions, high blood pressure. Treatment: Monitoring, medications (to lower blood pressure), surgery (if large or symptomatic). Prevention: Healthy diet, regular exercise, blood pressure control, avoiding smoking."
            },
            {
                "inputs": "### Instruction: What is aortic dissection? \n\n### Response: Aortic dissection is a serious condition in which the inner layer of the aorta tears, causing blood to flow between the layers of the aortic wall. Causes: High blood pressure, atherosclerosis, connective tissue disorders. Treatment: Emergency surgery, medications (to lower blood pressure). Prevention: Blood pressure control, healthy diet, regular exercise."
            },
            {
                "inputs": "### Instruction: Define pulmonary hypertension. \n\n### Response: Pulmonary hypertension is high blood pressure in the arteries that supply the lungs. The Causes are Chronic lung diseases, congenital heart defects, blood clots in the lungs. Treatment: Medications like (vasodilators, anticoagulants, diuretics), oxygen therapy, surgery in severe cases. Prevention: Managing underlying conditions, avoiding smoking, regular exercise."
            },
            {
                "inputs": "### Instruction: What is peripheral artery disease (PAD)? \n\n### Response: Peripheral artery disease (PAD) is a circulatory condition that causes narrowed arteries, reducing blood flow to the limbs. Causes: Atherosclerosis, diabetes, smoking, high blood pressure. Treatment: Lifestyle changes, medications (antiplatelets, statins), angioplasty, bypass surgery. Prevention: Healthy diet, regular exercise, smoking cessation, blood pressure control."
            },
            {
                "inputs": "### Instruction: Explain deep vein thrombosis (DVT). \n\n### Response: Deep vein thrombosis (DVT) is a blood clot that forms in a deep vein, usually in the legs. Causes: Prolonged immobility, surgery, pregnancy, obesity, smoking. Treatment: Blood thinners, compression stockings, surgery (in severe cases). Prevention: Moving regularly, staying hydrated, avoiding prolonged sitting."
            },
            {
                "inputs": "### Instruction: What is a pulmonary embolism? \n\n### Response: A pulmonary embolism is a blockage in one of the pulmonary arteries in the lungs, usually caused by a blood clot that travels from another part of the body. Causes: Deep vein thrombosis, surgery, prolonged immobility, obesity, smoking. Treatment: Blood thinners, oxygen therapy, surgery (in severe cases). Prevention: Moving regularly, staying hydrated, avoiding prolonged sitting."
            },
            {
                "inputs": "### Instruction: Define varicose veins. \n\n### Response: Varicose veins are enlarged, twisted veins that usually occur in the legs and can cause pain, swelling, and discomfort. Causes: Weak or damaged valves, pregnancy, obesity, prolonged standing. Treatment: Compression stockings, lifestyle changes, procedures (sclerotherapy, laser therapy). Prevention: Regular exercise, weight management, avoiding prolonged standing."
            },
            {
                "inputs": "### Instruction: What is a stroke? \n\n### Response: A stroke occurs when blood flow to part of the brain is interrupted, leading to brain cell damage. Causes: Ischemic stroke (blood clot), hemorrhagic stroke (ruptured blood vessel). Treatment: Ischemic stroke (clot-busting drugs, thrombectomy), hemorrhagic stroke (surgery, medications). Prevention: Healthy diet, regular exercise, blood pressure control, cholesterol management."
            },
            {
                "inputs": "### Instruction: What causes chest pain? \n\n### Response: Chest pain can result from various conditions, including heart problems, lung issues, or digestive disorders. Causes: Angina, heart attack, pericarditis, pulmonary embolism, gastroesophageal reflux disease (GERD). Treatment: Depends on the underlying cause. Prevention: Managing risk factors for heart disease, avoiding triggers for GERD."
            },
            {
                "inputs": "### Instruction: Define palpitations. \n\n### Response: Palpitations are feelings of having rapid, fluttering, or pounding heartbeats. Causes: Stress, anxiety, caffeine, nicotine, heart disease, electrolyte imbalances. Treatment: Addressing underlying causes, lifestyle changes, medications (if related to heart disease). Prevention: Avoiding triggers, stress management, regular exercise."
            },
            {
                "inputs": "### Instruction: What is dyspnea? \n\n### Response: Dyspnea, or shortness of breath, is the feeling of not being able to get enough air. Causes: Heart failure, asthma, chronic obstructive pulmonary disease (COPD), anemia. Treatment: Depends on the underlying cause. Prevention: Managing chronic conditions, avoiding smoking, maintaining a healthy weight."
            },
            {
                "inputs": "### Instruction: Explain shortness of breath. \n\n### Response: Shortness of breath, or dyspnea, is the sensation of being unable to breathe deeply or comfortably. Causes: Heart failure, asthma, chronic obstructive pulmonary disease (COPD), anemia. Treatment: Depends on the underlying cause. Prevention: Managing chronic conditions, avoiding smoking, maintaining a healthy weight."
            },
            {
                "inputs": "### Instruction: What causes fatigue? \n\n### Response: Fatigue is a feeling of extreme tiredness and lack of energy. Causes: Heart failure, anemia, sleep disorders, chronic fatigue syndrome. Treatment: Depends on the underlying cause. Prevention: Managing chronic conditions, healthy sleep habits, regular exercise."
            },
            {
                "inputs": "### Instruction: Define swelling. \n\n### Response: Swelling, or edema, is the buildup of fluid in the body's tissues, often seen in the legs, ankles, and feet. Causes: Heart failure, kidney disease, liver disease, venous insufficiency. Treatment: Medications (diuretics), compression stockings, addressing underlying cause. Prevention: Managing chronic conditions, avoiding prolonged sitting or standing."
            },
            {
                "inputs": "### Instruction: What is syncope? \n\n### Response: Syncope is a temporary loss of consciousness, commonly known as fainting. Causes: Low blood pressure, heart problems, dehydration, vasovagal response. Treatment: Addressing the underlying cause. Prevention: Staying hydrated, avoiding triggers, regular medical checkups."
            },
            {
                "inputs": "### Instruction: Explain fainting. \n\n### Response: Fainting, or syncope, is a sudden, brief loss of consciousness caused by decreased blood flow to the brain. Causes: Low blood pressure, heart problems, dehydration, vasovagal response. Treatment: Addressing the underlying cause. Prevention: Staying hydrated, avoiding triggers, regular medical checkups."
            },
            {
                "inputs": "### Instruction: What is cyanosis? \n\n### Response: Cyanosis is a bluish discoloration of the skin and mucous membranes, indicating low oxygen levels in the blood. Causes: Heart defects, lung diseases, blood disorders. Treatment: Depends on the underlying cause. Prevention: Managing chronic conditions, avoiding smoking, ensuring proper oxygenation."
            },
            {
                "inputs": "### Instruction: Define bluish discoloration of the skin. \n\n### Response: Bluish discoloration of the skin, or cyanosis, is a sign of low blood oxygen levels. Causes: Heart defects, lung diseases, blood disorders. Treatment: Depends on the underlying cause. Prevention: Managing chronic conditions, avoiding smoking, ensuring proper oxygenation."
            },
            {
                "inputs": "### Instruction: What is an electrocardiogram? \n\n### Response: An electrocardiogram (ECG or EKG) is a test that measures the electrical activity of the heart."
            },
            {
                "inputs": "### Instruction: Define echocardiogram. \n\n### Response: An echocardiogram is an ultrasound test that uses sound waves to create images of the heart."
            },
            {
                "inputs": "### Instruction: Explain cardiac catheterization. \n\n### Response: Cardiac catheterization is a procedure used to diagnose and treat certain cardiovascular conditions by inserting a catheter into the heart."
            },
            {
                "inputs": "### Instruction: What is an angiogram? \n\n### Response: An angiogram is an imaging test that uses X-rays to view the body's blood vessels."
            },
            {
                "inputs": "### Instruction: Define stress test. \n\n### Response: A stress test measures how the heart works during physical activity, typically involving exercise or medication to increase heart rate."
            },
            {
                "inputs": "### Instruction: What is a Holter monitor? \n\n### Response: A Holter monitor is a portable device that continuously records the heart's electrical activity for 24 to 48 hours."
            },
            {
                "inputs": "### Instruction: Explain angioplasty. \n\n### Response: Angioplasty is a procedure to restore blood flow through the artery by inflating a small balloon inside the blocked vessel."
            },
            {
                "inputs": "### Instruction: Define stent placement. \n\n### Response: Stent placement involves inserting a small mesh tube into a blocked artery to keep it open."
            },
            {
                "inputs": "### Instruction: What is coronary artery bypass grafting? \n\n### Response: Coronary artery bypass grafting (CABG) is a surgical procedure to improve blood flow to the heart by redirecting blood around blocked arteries."
            },
            {
                "inputs": "### Instruction: Define defibrillator. \n\n### Response: A defibrillator is a device that delivers an electric shock to the heart to restore a normal heartbeat."
            },
            {
                "inputs": "### Instruction: What is a pacemaker? \n\n### Response: A pacemaker is a device implanted in the chest to help control abnormal heart rhythms."
            },
            {
                "inputs": "### Instruction: Explain an implantable cardioverter-defibrillator. \n\n### Response: An implantable cardioverter-defibrillator (ICD) is a device that monitors heart rhythms and can deliver shocks to correct abnormal rhythms."
            },
            {
                "inputs": "### Instruction: What is thrombolytic therapy? \n\n### Response: Thrombolytic therapy involves using drugs to dissolve blood clots."
            },
            {
                "inputs": "### Instruction: Define anticoagulants. \n\n### Response: Anticoagulants are medications that help prevent blood clots."
            },
            {
                "inputs": "### Instruction: What are ACE inhibitors? \n\n### Response: ACE inhibitors are medications used to lower blood pressure by relaxing blood vessels."
            },
            {
                "inputs": "### Instruction: Define statins. \n\n### Response: Statins are drugs used to lower cholesterol levels in the blood."
            },
            {
                "inputs": "### Instruction: What are diuretics? \n\n### Response: Diuretics are medications that help remove excess fluid from the body by increasing urine production."
            },
            {
                "inputs": "### Instruction: Define valve replacement. \n\n### Response: Valve replacement is a surgery to replace a damaged heart valve with a prosthetic valve."
            },
            {
                "inputs": "### Instruction: Explain hyperlipidemia. \n\n### Response: Hyperlipidemia is the condition of having high levels of lipids (fats) in the blood. Causes: Poor diet, genetic factors, obesity, sedentary lifestyle. Treatment: Medications (statins, fibrates), lifestyle changes. Prevention: Healthy diet, regular exercise, weight management."
            },
            {
                "inputs": "### Instruction: What is high cholesterol? \n\n### Response: High cholesterol is a condition where there are high levels of cholesterol in the blood, increasing the risk of heart disease. Causes: Poor diet, genetic factors, obesity, sedentary lifestyle. Treatment: Medications (statins, fibrates), lifestyle changes. Prevention: Healthy diet, regular exercise, weight management."
            },
            {
                "inputs": "### Instruction: Define diabetes. \n\n### Response: Diabetes is a chronic condition characterized by high blood sugar levels due to the body's inability to produce or use insulin effectively. Causes: Genetic factors, poor diet, obesity, lack of exercise. Treatment: Medications (insulin, metformin), lifestyle changes, blood sugar monitoring. Prevention: Healthy diet, regular exercise, weight management."
            },
            {
                "inputs": "### What is chest pain type ASY? \n\n### Response: ASY or Asymptomatic means the absence of chest pain but hintsat potential heart issues."
            },
            {
                "inputs": "### What is chest pain type ATA? \n\n### Response: ATA or atypical chest pain refers to chest discomfort that does not fit the typical pattern of angina."
            },
            {
                "inputs": "### What is chest pain type NAP? \n\n### Response: NAP or Non-Anginal Pain refers to chest pain that is not related to the heart."
            },
            {
                "inputs": "### What is chest pain type TA? \n\n### Response: TA or Typical Angina is the common heart-related chest pain"
            },
            {
                "inputs": "### What could cause chest pain? \n\n### Response: Chest pain can stem from a heart problem, but other possible causes include a lung infection, muscle strain, a rib injury, or a panic attack. Some of these are serious conditions and need medical attention."
            },
            {
                "inputs": "### What is chest pain? \n\n### Response: Chest pain is discomfort or pain that you feel anywhere along the front of your body between your neck and upper abdomen. Symptoms of a possible heart attack include chest pain and pain that radiates down the shoulder and arm. Some people (older adults, people with diabetes, and women) may have little or no chest pain."
            },
            {
                "inputs": "### What Myocarditis? \n\n### Response: Myocarditis is inflammation of the heart muscle, resulting in symptoms that are similar to a heart attack, such as: chestpain, shortness of breath and fast or irregular heartbeat. Accordign to the British Heart Foundation,myocarditis usuallu results form a viral infection."
            },
            {
                "inputs": "### How does Angina feels like? \n\n### Response: Angina feels like a squeezing pain or pressure on the chest. It occursTrusted Source when not enough blood is getting to the heart. A person may also feel pain in the: shoulder, back, neck, jaw, arms. Also, angina can feel like indigestion and it is the most common symptom of coronary artery disease."
            },
            {
                "inputs": "### Waht is Pleurisy ? \n\n### Response: Pleurisy is inflammation of the membrane that covers the lungs. Its symptoms include: chest or shoulder pain, pain is worth when breathing, coughing,sneezing, or moving the trunk or chest wall. Also pain may be dull, aching, or “catching. Without treatment, it can lead to life threatening complications."
            },
            {
                "inputs": "### What is Pneumonia? \n\n### Response: Lung infections such as pneumonia can cause sharp or stabbing chest pain, especially when breathing deeply or coughing."
            },
            {
                "inputs": "### What are the symptoms of Pneumonia? \n\n### Response: symptoms of pneumonia inclclude: fever, sweating, and chills. The patient may cough up phlegm, colored green, yellow, or containing blood. They are likely to have shortness of breath, bluish tinge to the lips or fingetips, rapid, shallow breathing,low appetite, low energy, and fatigue."
            },
            {
                "inputs": "### What are the symptoms of Pneumonia in elderly people? \n\n### Response: pneumonia can cause confusion in older people.A person with breathing difficulty ,no matter younf or olde, needs immediate medical attention, as pneumonia can be life threatening."
            },
            {
                "inputs": "### Does COVID-19 cause respiratory symptoms? \n\n### Response: A person with COVID-19 may experience respiratory symptoms, pain, or pressure in the chest. If you are experiencing severe covid-19 symptoms such as persistent pain or pressure in the chest, breathing difficulty, blue lips or nails and you have difficulty staying awake, seek medical attention immediately."
            },
            {
                "inputs": "### What is Tuberculosis infection? \n\n### Response: Tuberculosis (TB) is a bacterial infection that usually affects the lungs. It can cause chest pain and a bad cough, which may bring up blood or sputum. Also it can cause weigth loss and a fever or night sweats."
            },
            {
                "inputs": "### What does collapsed lung mean? \n\n### Response: When air builds up in the space between the lungs and ribs, it leads to a collapsed lung, also known as pneumothorax. some people have no symptoms but some symptoms like discomfort when breating, faster breathing rate, swelling on one side of the chest and reduced breathing sounds may occur."
            },
            {
                "inputs": "### What does Panic attack mean? \n\n### Response: A panic attack is a sudden attack of panic or fear. Often a person does not know why it happens, but it may be a symptom of a condition known as panic disorder."
            },
            {
                "inputs": "### What is Costochondritis? \n\n### Response: Costochondritis is inflammation of the cartilage of the rib cage. It can cause pain and tenderness in the chest. The pain may start suddenly. Costochondritis pain may get worse when: lying down, doing exercise or moving the upper body, breathing deeply and coughing or sneezing."
            },
            {
                "inputs": "### What can cause chest pain besides a heart attack? \n\n### Response: Some common causes of chest pain, besides a heart attack, include myocarditis, angina, pneumonia, and Covid-19."
            },
            {
                "inputs": "### What is an anaphylactic shock? \n\n### Response: an extreme allergic reaction that usually involves heart failure, circulatory collapse, a severe asthma-like difficulty in breathing and sometimes results in death."
            },
            {
                "inputs": "### What is atropine? \n\n### Response: Atropine is a drug used to increase the heart rate"
            },
            {
                "inputs": "### What is bradycardic? \n\n### Response: bradycardic  is a slowing of the heart rate to less than 50 beats per minute."
            },
            {
                "inputs": "### What is cardiac tamponade ? \n\n### Response: compression of the heart from fluid such as an effusion or blood."
            },
            {
                "inputs": "### What is CHF short for? \n\n### Response: CHF is abbreviation for congestive heart failure."
            },
            {
                "inputs": "### What is CPK short for? \n\n### Response: creatine phosphokinase, an enzyme that elevates in the blood when a heart attack occurs, used as a confirmation of a heart attack and as a gauge of damage."
            },
            {
                "inputs": "### What does diastolic mean? \n\n### Response: it is the pressure during the relaxing of the heart."
            },
            {
                "inputs": "### What is digitalis? \n\n### Response: it is a drug prescribed for congestive heart failure."
            },
            {
                "inputs": "### What is tPA? \n\n### Response: This is the abbreviation for tissue plasminogen activator, a drug used as an alternative to angioplasty to break up blood clots during a heart attack."
            },
            {
                "inputs": "### Instruction: What does a complete blood count (CBC) test measure? \n\n### Response: A complete blood count (CBC) test measures the levels of different components in your blood, including red blood cells, white blood cells, hemoglobin, hematocrit, and platelets."},
            {
                "inputs": "### Instruction: How can you prevent osteoporosis? \n\n### Response: Osteoporosis can be prevented through a diet rich in calcium and vitamin D, regular weight-bearing exercise, and avoiding smoking and excessive alcohol consumption."},
            {
                "inputs": "### Instruction: What is the function of insulin in the body? \n\n### Response: Insulin is a hormone that helps regulate blood glucose levels by allowing cells to take in glucose to be used for energy or stored for future use."},
            {
                "inputs": "### Instruction: What is chronic obstructive pulmonary disease (COPD)? \n\n### Response: Chronic obstructive pulmonary disease (COPD) is a group of lung conditions that cause breathing difficulties, including emphysema and chronic bronchitis."},
            {
                "inputs": "### Instruction: How is a urinary tract infection (UTI) treated? \n\n### Response: A urinary tract infection (UTI) is typically treated with antibiotics, and it's important to drink plenty of water to help flush out the bacteria."},
            {
                "inputs": "### Instruction: What are the risk factors for developing cardiovascular disease? \n\n### Response: Risk factors for cardiovascular disease include high blood pressure, high cholesterol, smoking, obesity, physical inactivity, and diabetes."},
            {
                "inputs": "### Instruction: What is cardiovascular disease? \n\n### Response:Cardiovascular disease refers to a group of disorders affecting the heart and blood vessels, including coronary artery disease, heart failure, and arrhythmias."},
            {
                "inputs": "### Instruction: What is the difference between a heart attack and cardiac arrest? \n\n### Response:A heart attack occurs when blood flow to part of the heart is blocked, while cardiac arrest is when the heart suddenly stops beating, often due to an electrical disturbance."},
            {
                "inputs": "### Instruction: How does one perform CPR? \n\n### Response: To perform CPR, place your hands on the center of the person's chest and press down hard and fast at a rate of 100-120 compressions per minute, and give rescue breaths if trained."},
            {
                "inputs": "### Instruction: What is anemia? \n\n### Response: Anemia is a condition where you don't have enough healthy red blood cells to carry adequate oxygen to your body's tissues, often resulting in fatigue and weakness."},
            {
                "inputs": "### Instruction: What are the benefits of regular exercise? \n\n### Response: Regular exercise can help control weight, reduce the risk of chronic diseases, improve mental health and mood, and strengthen bones and muscles."},
            {
                "inputs": "### Instruction: What is asthma? \n\n### Response: Asthma is a chronic lung disease that inflames and narrows the airways, causing episodes of wheezing, breathlessness, chest tightness, and coughing."},
            {
                "inputs": "### Instruction: How is depression treated? \n\n### Response: Depression is often treated with a combination of medications, such as antidepressants, and therapy, including cognitive-behavioral therapy and counseling."},
            {
                "inputs": "### Instruction: What are the symptoms of hypothyroidism? \n\n### Response: Symptoms of hypothyroidism can include fatigue, weight gain, cold intolerance, dry skin, and hair loss."},
            {
                "inputs": "### Instruction: What is heart disease? \n\n### Response: Heart disease encompasses various conditions affecting the heart, including coronary artery disease, arrhythmias, and heart defects. Causes: High blood pressure, high cholesterol, smoking, diabetes, obesity, sedentary lifestyle, stress, and family history. Prevention: Healthy diet, regular exercise, smoking cessation, blood pressure control, cholesterol management."
            },
            {
                "inputs": "### Instruction: Define cardiac. \n\n### Response: Cardiac relates to anything associated with the heart."
            },
            {
                "inputs": "### Instruction: What is cardiology? \n\n### Response: Cardiology is the branch of medicine that deals with the diagnosis and treatment of heart conditions."
            },
            {
                "inputs": "### Instruction: Who is a cardiologist? \n\n### Response: A cardiologist is a doctor who specializes in diagnosing and treating heart diseases."
            },
            {
                "inputs": "### Instruction: Explain cardiomyopathy. \n\n### Response: Cardiomyopathy is a disease of the heart muscle that affects its size, shape, and structure. Causes: Genetic factors, long-term high blood pressure, heart tissue damage from a heart attack, chronic rapid heart rate, metabolic disorders. Treatment: Medications (beta-blockers, ACE inhibitors, diuretics), lifestyle changes, implanted devices, surgery."
            },
            {
                "inputs": "### Instruction: Describe coronary artery disease. \n\n### Response: Coronary artery disease is the narrowing or blockage of the coronary arteries, usually caused by atherosclerosis. Causes: High cholesterol, high blood pressure, smoking, diabetes, sedentary lifestyle, unhealthy diet. Treatment: Lifestyle changes, medications (statins, beta-blockers, nitroglycerin), angioplasty, stent placement, coronary artery bypass grafting."
            },
            {
                "inputs": "### Instruction: What is myocardial infarction? \n\n### Response: Myocardial infarction, commonly known as a heart attack, occurs when blood flow to a part of the heart is blocked. Causes: Coronary artery disease, blood clot. Treatment: Thrombolytic therapy, angioplasty, stent placement, medications (anticoagulants, beta-blockers, ACE inhibitors). Prevention: Healthy diet, regular exercise, smoking cessation, cholesterol management, blood pressure control."
            },
            {
                "inputs": "### Instruction: Explain heart attack. \n\n### Response: A heart attack, or myocardial infarction, happens when a part of the heart muscle doesn't receive enough blood. Causes: Coronary artery disease, blood clot. Treatment: Thrombolytic therapy, angioplasty, stent placement, medications (anticoagulants, beta-blockers, ACE inhibitors). Prevention: Healthy diet, regular exercise, smoking cessation, cholesterol management, blood pressure control."
            },
            {
                "inputs": "### Instruction: Define angina. \n\n### Response: Angina is chest pain caused by reduced blood flow to the heart muscles. Causes: Coronary artery disease, stress, physical exertion. Treatment: Medications (nitroglycerin, beta-blockers, calcium channel blockers), lifestyle changes, angioplasty, stent placement. Prevention: Healthy diet, regular exercise, stress management, smoking cessation."
            },
            {
                "inputs": "### Instruction: What is arrhythmia? \n\n### Response: Arrhythmia is an irregular heartbeat, which can be too fast, too slow, or erratic. Causes: Heart disease, high blood pressure, diabetes, smoking, excessive alcohol consumption, stress. Treatment: Medications (beta-blockers, anti-arrhythmic drugs), lifestyle changes, implanted devices (pacemaker, cardioverter-defibrillator), ablation therapy."
            },
            {
                "inputs": "### Instruction: Explain atrial fibrillation. \n\n### Response: Atrial fibrillation is a type of arrhythmia where the upper chambers of the heart beat irregularly. Causes: High blood pressure, heart attack, coronary artery disease, abnormal heart valves, congenital heart defects. Treatment: Medications (anticoagulants, beta-blockers, anti-arrhythmic drugs), electrical cardioversion, ablation therapy, lifestyle changes."
            },
            {
                "inputs": "### Instruction: What is heart failure? \n\n### Response: Heart failure occurs when the heart cannot pump enough blood to meet the body's needs. Causes: Coronary artery disease, high blood pressure, previous heart attack, cardiomyopathy. Treatment: Medications (diuretics, ACE inhibitors, beta-blockers), lifestyle changes, implanted devices, surgery. Prevention: Healthy diet, regular exercise, blood pressure control, cholesterol management."
            },
            {
                "inputs": "### Instruction: Define congestive heart failure. \n\n### Response: Congestive heart failure is a condition where the heart's ability to pump blood is inadequate, causing fluid buildup in the body. Causes: Coronary artery disease, high blood pressure, previous heart attack, cardiomyopathy. Treatment: Medications (diuretics, ACE inhibitors, beta-blockers), lifestyle changes, implanted devices, surgery. Prevention: Healthy diet, regular exercise, blood pressure control, cholesterol management."
            },
            {
                "inputs": "### Instruction: How does smoking affect health? \n\n### Response: Smoking is a major risk factor for many diseases, including cardiovascular disease, as it damages the blood vessels and heart. Causes: Tobacco use. Treatment: Smoking cessation programs, medications (nicotine replacement therapy, bupropion, varenicline). Prevention: Avoiding smoking, promoting smoke-free environments."
            },
            {
                "inputs": "### Instruction: Explain obesity. \n\n### Response: Obesity is a condition characterized by excessive body fat, increasing the risk of various health problems, including heart disease. Causes: Poor diet, lack of physical activity, genetic factors. Treatment: Lifestyle changes, medications (weight loss drugs), surgery (bariatric surgery). Prevention: Healthy diet, regular exercise, weight management."
            },
            {
                "inputs": "### Instruction: What is a sedentary lifestyle? \n\n### Response: A sedentary lifestyle involves little physical activity, contributing to various health issues, including cardiovascular disease. Causes: Lack of exercise, prolonged sitting or inactivity. Prevention: Regular physical activity, reducing sedentary behavior, promoting active lifestyles."
            },
            {
                "inputs": "### Instruction: Define family history. \n\n### Response: Family history refers to the presence of certain diseases in immediate family members, which can increase one's risk of developing those conditions."
            },
            {
                "inputs": "### Instruction: How does stress affect health? \n\n### Response: Stress is a physical and emotional response to external pressures, which can negatively impact heart health. Causes: Work pressure, personal issues, financial problems. Treatment: Stress management techniques (meditation, exercise, counseling), medications (if necessary). Prevention: Regular exercise, healthy work-life balance, relaxation techniques."
            },
            {
                "inputs": "### Instruction: What are the effects of alcohol consumption on health? \n\n### Response: Excessive alcohol consumption can increase the risk of heart disease and other health problems. Causes: Drinking large amounts of alcohol regularly. Treatment: Alcohol reduction programs, counseling, medications (if necessary). Prevention: Moderation in alcohol consumption, promoting awareness of risks."
            },
            {
                "inputs": "### Instruction: What is a healthy diet? \n\n### Response: A healthy diet includes a balance of nutrients that support overall health and can help prevent heart disease."
            },
            {
                "inputs": "### Instruction: Explain blood pressure control. \n\n### Response: Blood pressure control involves maintaining blood pressure within a healthy range to reduce the risk of heart disease."
            },
            {
                "inputs": "### Instruction: What is cholesterol management? \n\n### Response: Cholesterol management involves maintaining healthy cholesterol levels through diet, exercise, and medication if needed."
            },
            {
                "inputs": "### Instruction: Define smoking cessation. \n\n### Response: Smoking cessation is the process of quitting smoking, which significantly reduces the risk of heart disease and other health issues."
            },
            {
                "inputs": "### Instruction: What are the atria? \n\n### Response: The atria are the two upper chambers of the heart that receive blood from the body and lungs."
            },
            {
                "inputs": "### Instruction: Define ventricles. \n\n### Response: The ventricles are the two lower chambers of the heart that pump blood to the body and lungs."
            },
            {
                "inputs": "### Instruction: What is the aorta? \n\n### Response: The aorta is the main artery that carries oxygen-rich blood from the heart to the rest of the body."
            },
            {
                "inputs": "### Instruction: Explain the pulmonary artery. \n\n### Response: The pulmonary artery carries deoxygenated blood from the right ventricle of the heart to the lungs."
            },
            {
                "inputs": "### Instruction: What are coronary arteries? \n\n### Response: The coronary arteries supply blood to the heart muscle itself."
            },
            {
                "inputs": "### Instruction: Define the mitral valve. \n\n### Response: The mitral valve is located between the left atrium and left ventricle, preventing blood from flowing backward into the atrium."
            },
            {
                "inputs": "### Instruction: What is the aortic valve? \n\n### Response: The aortic valve is located between the left ventricle and the aorta, allowing blood to flow from the heart to the body."
            },
            {
                "inputs": "### Instruction: Explain the tricuspid valve. \n\n### Response: The tricuspid valve is located between the right atrium and right ventricle, preventing blood from flowing backward into the atrium."
            },
            {
                "inputs": "### Instruction: What is the pulmonary valve? \n\n### Response: The pulmonary valve is located between the right ventricle and the pulmonary artery, allowing blood to flow to the lungs."
            },
            {
                "inputs": "### Instruction: Define the pericardium. \n\n### Response: The pericardium is a double-walled sac that surrounds and protects the heart."
            },
            {
                "inputs": "### Instruction: What is the myocardium? \n\n### Response: The myocardium is the muscular tissue of the heart."
            },
            {
                "inputs": "### Instruction: Explain the endocardium. \n\n### Response: The endocardium is the inner lining of the heart chambers and valves."
            },
            {
                "inputs": "### Instruction: What is a myocardial infarction? \n\n### Response: Myocardial infarction, commonly known as a heart attack, is a blockage of blood flow to the heart muscle. Causes: Coronary artery disease, blood clot. Symptoms: Chest pain, shortness of breath, nausea, sweating. Treatment: Medications (aspirin, thrombolytics, nitroglycerin), coronary angioplasty, lifestyle changes. Prevention: Healthy diet, regular exercise, smoking cessation, managing risk factors such as high blood pressure and high cholesterol."
            },
            {
                "inputs": "### Instruction: Define arrhythmia. \n\n### Response: Arrhythmia is an irregular heartbeat, which can be too fast, too slow, or erratic. Causes: Heart disease, high blood pressure, diabetes, smoking, excessive alcohol consumption, stress. Symptoms: Palpitations, dizziness, shortness of breath, chest pain. Treatment: Medications (beta-blockers, anti-arrhythmic drugs), lifestyle changes, implanted devices (pacemaker, cardioverter-defibrillator), ablation therapy. Prevention: Healthy lifestyle, managing heart disease and risk factors."
            },
            {
                "inputs": "### Instruction: Explain atherosclerosis. \n\n### Response: Atherosclerosis is a condition where the arteries become narrowed and hardened due to a buildup of plaque. Causes: High cholesterol, high blood pressure, smoking, diabetes. Symptoms: May be asymptomatic until an artery is significantly blocked or ruptures, leading to chest pain, shortness of breath, heart attack, or stroke. Treatment: Medications (statins, antiplatelet agents), lifestyle changes, surgical procedures (angioplasty, stent placement, bypass surgery). Prevention: Healthy diet, regular exercise, smoking cessation, cholesterol and blood pressure management."
            },
            {
                "inputs": "### Instruction: What is hypertrophic cardiomyopathy? \n\n### Response: Hypertrophic cardiomyopathy is a condition in which the heart muscle becomes abnormally thick due to genetic factors. Causes: Genetic mutations. Symptoms: Shortness of breath, chest pain, palpitations, fainting. Treatment: Medications (beta-blockers, calcium channel blockers), lifestyle changes, surgical procedures (septal myectomy, alcohol septal ablation), implanted devices (pacemaker, cardioverter-defibrillator). Prevention: Regular medical checkups, managing symptoms and complications."
            },
            {
                "inputs": "### Instruction: What are the symptoms of hypertrophic cardiomyopathy? \n\n### Response: Symptoms of hypertrophic cardiomyopathy include: chest pain, changes in heart rhythm, shortness of breath, lightheadness, dizziness, fainting, and swelling in the feet, ankles, legs, and abdomen.A person who notices this should seek medical advice, as it can worsen over time and lead to other complications, such as blood clots and heart failure."
            },
            {
                "inputs": "### Instruction: Define hyperthyroidism. \n\n### Response: Hyperthyroidism is the overactivity of the thyroid gland, leading to a rapid heartbeat and an increased rate of metabolism. Causes: Graves' disease, thyroid nodules, thyroiditis. Symptoms: Weight loss, rapid heartbeat, nervousness, irritability, sweating. Treatment: Medications (antithyroid drugs, beta-blockers), radioactive iodine therapy, surgery. Prevention: Regular monitoring and management of thyroid levels, early treatment of thyroid conditions."
            },
            {
                "inputs": "### Instruction: Explain hypothyroidism. \n\n### Response: Hypothyroidism is the underactivity of the thyroid gland, causing fatigue and weight gain. Causes: Hashimoto's thyroiditis, iodine deficiency, certain medications. Symptoms: Fatigue, weight gain, cold intolerance, dry skin, hair loss. Treatment: Thyroid hormone replacement therapy. Prevention: Regular monitoring and management of thyroid levels, adequate iodine intake."
            },
            {
                "inputs": "### Instruction: What is Cushing's syndrome? \n\n### Response: Cushing's syndrome is a condition that occurs from exposure to high cortisol levels for a long time. Causes: Long-term use of corticosteroid medications, pituitary adenomas, adrenal tumors. Symptoms: Weight gain, especially around the abdomen and face, thinning skin, easy bruising, high blood pressure, osteoporosis. Treatment: Reducing corticosteroid use, surgery, radiation therapy, medications to control cortisol production. Prevention: Monitoring and managing corticosteroid use, early treatment of underlying conditions."
            },
            {
                "inputs": "### Instruction: Define diabetic ketoacidosis. \n\n### Response: Diabetic ketoacidosis (DKA) is a serious diabetes complication where the body produces excess blood acids (ketones). Causes: Insufficient insulin levels, infection, stress, missed insulin doses. Symptoms: Excessive thirst, frequent urination, nausea, abdominal pain, shortness of breath, confusion. Treatment: Fluid replacement, electrolyte replacement, insulin therapy. Prevention: Regular blood sugar monitoring, proper insulin use, managing diabetes effectively."
            },
            {
                "inputs": "### Instruction: What are the symptoms of a heart attack? \n\n### Response: Symptoms of a heart attack can include chest pain or discomfort, shortness of breath, nausea, lightheadedness, and pain in the arms, neck, jaw, or back."
            },
            {
                "inputs": "### How is coronary artery disease diagnosed? \n\n### Response: Coronary artery disease is diagnosed using tests such as electrocardiograms (ECG), stress tests, echocardiograms, and coronary angiograms."
            },
            {
                "inputs": "### What are the signs of heart failure? \n\n### Response: Signs of heart failure include shortness of breath, fatigue, swollen legs, rapid or irregular heartbeat, and persistent cough or wheezing."
            },
            {
                "inputs": "### How is hypertension treated? \n\n### Response: Hypertension is treated with lifestyle changes such as diet and exercise, and medications including diuretics, ACE inhibitors, and beta-blockers."
            },
            {
                "inputs": "### What is angioplasty? \n\n### Response: Angioplasty is a procedure to open narrowed or blocked coronary arteries using a balloon catheter. Often, a stent is placed to keep the artery open."
            },
            {
                "inputs": "### What are the treatment options for atrial fibrillation? \n\n### Response: Treatment options for atrial fibrillation include medications to control heart rate or rhythm, electrical cardioversion, catheter ablation, and sometimes surgery."
            },
            {
                "inputs": "### How can I reduce my risk of heart disease? \n\n### Response: To reduce your risk of heart disease, you should maintain a healthy diet, exercise regularly, avoid smoking, limit alcohol intake, and manage stress and health conditions like hypertension and diabetes."
            },
            {
                "inputs": "### What lifestyle changes can help manage high cholesterol? \n\n### Response: Managing high cholesterol involves eating a healthy diet low in saturated fats, exercising regularly, losing weight, and possibly taking medications prescribed by a doctor."
            },
            {
                "inputs": "### What is angina and how is it treated? \n\n### Response: Angina is chest pain caused by reduced blood flow to the heart. It is treated with medications, lifestyle changes, and sometimes procedures like angioplasty or coronary artery bypass surgery."
            },
            {
                "inputs": "### What is myocarditis? \n\n### Response: Myocarditis is inflammation of the heart muscle, often caused by viral infections. Symptoms can include chest pain, fatigue, shortness of breath, and arrhythmias. Treatment focuses on managing symptoms and underlying causes."
            },
            {
                "inputs": "### What are the symptoms and treatment options for pericarditis? \n\n### Response: Pericarditis is inflammation of the pericardium, causing sharp chest pain, fever, and a pericardial friction rub. Treatment includes anti-inflammatory medications and, in severe cases, surgery."
            },
            {
                "inputs": "### How is blood pressure monitored and why is it important? \n\n### Response: Blood pressure is monitored using a sphygmomanometer. Regular monitoring helps detect hypertension early, manage existing conditions, and prevent complications like heart disease and stroke."
            },
            {
                "inputs": "### What is the role of beta-blockers in treating heart conditions? \n\n### Response: Beta-blockers reduce the heart's workload and help it beat more regularly, making them useful in treating high blood pressure, angina, heart failure, and arrhythmias."
            },
            {
                "inputs": "### What should I do if I think someone is having a heart attack? \n\n### Response: If you think someone is having a heart attack, call emergency services immediately, have them sit down and rest, and if available, give them aspirin unless they are allergic."
            },
            {
                "inputs": "### How should we perform CPR on someone in cardiac arrest? \n\n### Response: To perform CPR, place your hands on the center of the person's chest and push hard and fast at a rate of 100-120 compressions per minute. If trained, give rescue breaths in between compressions."
            },
            {
                "inputs": "### What are the risk factors for developing cardiovascular disease? \n\n### Response: Risk factors include high blood pressure, high cholesterol, smoking, obesity, physical inactivity, diabetes, and a family history of heart disease."
            }
        ]
        # SPLITING DATA
        batch_size = 100
        sample_batches = [samples[i:i + batch_size] for i in range(0, len(samples), batch_size)]
        # NUM EPOCHS 3,4,5 TEST
        num_epochs = 3
        count = 0
        while count < num_epochs:
            print(f"Fine-tuning the model, iteration {count + 1}")
            for batch in sample_batches:
                new_model_adapter.fine_tune(samples=batch)
            count = count + 1
        # AFTER FINE-TUNING
        completion = new_model_adapter.complete(query=sample_query, max_generated_token_count=100).generated_output
        print(f"Generated (after fine-tune): {completion}")


@app.route('/', methods=['POST'])
def ask():
    data = request.get_json()
    query = data['query']
    completion = new_model_adapter.complete(query=query, max_generated_token_count=100).generated_output
    return jsonify({"response": completion})

if __name__ == '__main__':
    fine_tune_model()
    #RUN ON LOCAL SERVER
    app.run(host='0.0.0.0', port=5000)
    #main()