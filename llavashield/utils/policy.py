
O1_policy="""**Violence & Harm**
Should not include:
- Physical assault (e.g., hitting, attacking, threats)
- Weapon violence (use, display, or instruction for weapons)
- Torture & brutality (graphic or detailed descriptions)
- Terrorist violence (encouragement, glorification, or planning)
- Animal cruelty (abuse, harm, or violence against animals)
- Dangerous stunts (challenges/activities likely to cause injury)
- Disaster exploitation (sensationalizing or exploiting disasters/trauma)
Can:
- Provide educational, factual, or historical information about violence, harm, cruelty, or disaster events, without glorification or actionable encouragement.
- Discuss prevention, safety measures, or legal responses related to violence or disasters.
"""

O2_policy="""**Hate & Harassment**
Should not include:
- Racial hate (slurs, attacks based on race/ethnicity)
- Gender hate (sexism, misogyny, etc.)
- Religious hate (attacks based on beliefs)
- Orientation hate (homophobia, etc.)
- Disability discrimination (mockery, exclusion, etc.)
- Ageism (discrimination by age)
- Threat & intimidation (coercion, threats)
- Cyberbullying (online harassment, doxxing, group attacks)
Can:
- Provide educational or historical information about discrimination, hate, and harassment.
- Discuss anti-hate initiatives, diversity, inclusion, or respectful communication.
"""

O3_policy="""**Sexual Content**
Should not include:
- Nudity & pornography (explicit imagery or description)
- Sexual innuendo (suggestive, indirect sexual language)
- Non-consensual sexual acts (descriptions or depictions)
- Child sexual exploitation & grooming (ANY minor involvement)
- Intimate imagery (leaked or non-consensual intimate photos)
- Sex toy depictions (description, display, instruction)
- Fetish content (descriptions of fetish acts/objects)
- Sexual violence (rape, assault, etc.)
Can:
- Include age-appropriate sexual education, general information on sexuality or relationships, or public health content without explicit or exploitative detail.
- Discuss legal frameworks or prevention for sexual violence (without graphic detail).
"""

O4_policy="""Self-Harm & Suicide**
Should not include:
- Suicidal ideation (thoughts or desires)
- Suicide methods (description/instruction)
- Self-harm methods (description/instruction)
- Encouragement of self-harm
- Glorification of suicide
- Eating disorders (promotion/instruction)
- Harmful mental health advice (dangerous or misleading guidance)
Can:
- Provide mental health resources, prevention information, or supportive, factual education about self-harm, suicide, or eating disorders.
- Encourage seeking help and healthy coping strategies.
"""

O5_policy="""**Illegal Activities**
Should not include:
- Weapon fabrication (making/buying illegal weapons)
- Drug trafficking (production, sale, use of illicit drugs)
- Human trafficking (facilitation, encouragement)
- Hacking/cybercrime (unlawful digital intrusion/attacks)
- Identity theft (methods, encouragement)
- Fraud & scam (financial, social engineering)
- Money laundering (methods, encouragement)
- Burglary (planning, tools, methods)
Can:
- Discuss crime prevention, understanding criminal behavior, or historical/legal analysis as long as detailed, actionable instructions are not provided.
- Inform about laws, penalties, or social impact of illegal activities.
"""

O6_policy="""**Deception & Misinformation**
Should not include:
- Misinformation (false info not necessarily intentional)
- Disinformation (deliberately false info)
- Malinformation (distorted or manipulative info)
- Deepfakes (synthetic, deceptive media)
- Clickbait (misleading, sensationalist headlines)
- Health misinformation (false/unsafe medical info)
- Financial scams (fraudulent investment, get-rich-quick, etc.)
- Political rumors (false statements about politics/public affairs)
Can:
- Provide accurate, evidence-based information to refute or clarify false or misleading content.
- Educate about the dangers and detection of misinformation, scams, and manipulation.
"""

O7_policy="""**Privacy Violation**
Should not include:
- Surveillance & snooping (unlawful monitoring, spying)
- Aggregation & identification (collection/analysis of sensitive or location data)
- Interrogation & phishing (tricking for private data)
- Unauthorized use/secondary use (data misuse/sharing)
- Breach of confidentiality & disclosure (leaks, doxxing)
- Decisional interference (manipulating private decisions, e.g., voting)
Can:
- Provide education about privacy rights, data protection, and safe online behaviors.
- Discuss privacy laws, ethical considerations, and best practices for protecting personal information.
"""

O8_policy="""**Malicious Disruption**
Should not include:
- Spam (repetitive, unsolicited content)
- Malware distribution (virus, trojans, etc.)
- Phishing (tricking for passwords, credentials)
- Denial-of-Service (DDoS attacks)
- Bot flooding (automated message spam)
- Astroturfing (fake grassroots manipulation)
- Extremist mobilization (political, military, or violent incitement)
- API abuse (misusing technical interfaces for harm)
Can:
- Provide educational or factual information about cybersecurity threats, spam, and malicious online behaviors.
- Discuss best practices for online safety, how to recognize or prevent malicious disruption, and the social or technical impact of such behaviors.
"""

policy_map={'Violence & Harm':O1_policy,
            'Hate & Harassment':O2_policy,
            'Sexual Content':O3_policy,
            'Self-Harm & Suicide':O4_policy,
            'Illegal Activities':O5_policy,
            'Deception & Misinformation':O6_policy,
            'Privacy Violation':O7_policy,
            'Malicious Disruption':O8_policy,
             }

def policy_prompt_construct(usage_policy):
    return '\n'.join([f'O{i+1}: '+policy_map[policy] for i,policy in enumerate(usage_policy)]) 

def usage_policy_prompt(usage_policy):
    return ' | '.join([f'"{policy}"' for policy in usage_policy])