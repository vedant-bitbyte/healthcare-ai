"""
Prompt templates for different chunk categories to ensure diversity in instruction generation.
"""

PROMPT_TEMPLATES = {
    "Policy": [
        "Explain the key policy details described in the following text.",
        "Summarize the policy guidelines mentioned in this excerpt.",
        "What are the main benefits and implementation strategies outlined here?",
        "Provide an overview of the regulatory framework discussed in this document.",
        "Detail the policy recommendations provided in the text."
    ],
    "Statistics": [
        "Explain the key statistics and figures presented in this text.",
        "What are the numerical findings detailed in this document?",
        "Interpret the trends based on the data provided below.",
        "Summarize the rates and ratios mentioned in the excerpt.",
        "Compare the numbers and highlight the key statistical insights."
    ],
    "Healthcare Infrastructure": [
        "Describe the healthcare infrastructure mentioned in this section.",
        "What does the text say about hospital and clinic facilities?",
        "Summarize the details regarding bed capacity and medical infrastructure.",
        "Explain the infrastructure gaps or improvements discussed here.",
        "Provide an overview of the physical healthcare facilities described."
    ],
    "Disease Burden": [
        "What information is provided regarding disease burden in this text?",
        "Summarize the morbidity and mortality data presented.",
        "Explain the impact of the diseases mentioned in this excerpt.",
        "What are the key findings related to infections and health conditions?",
        "Describe the public health challenges highlighted by this disease data."
    ],
    "Maternal Health": [
        "What does the text indicate about maternal health and pregnancy?",
        "Summarize the antenatal and neonatal care guidelines or data provided.",
        "Explain the challenges and recommendations regarding childbirth mentioned.",
        "What are the government initiatives for maternal health described here?",
        "Detail the maternal health improvements outlined in the excerpt."
    ],
    "Healthcare Workforce": [
        "Discuss the state of the healthcare workforce as described in this text.",
        "What does the document reveal about doctor and nurse availability?",
        "Explain the workforce shortages or distribution issues mentioned.",
        "Summarize the details regarding medical staff and specialists.",
        "What are the key takeaways regarding human resources in health from this excerpt?"
    ],
    "Budget": [
        "Explain the financial allocations or budget details provided.",
        "Summarize the healthcare expenditure and funding mentioned in the text.",
        "What does this excerpt reveal about health finance and costs?",
        "Detail the budgetary provisions and crore allocations discussed.",
        "Provide an overview of the financial health data presented."
    ],
    "Recommendation": [
        "What are the main recommendations suggested in this text?",
        "Summarize the proposed actions or improvements mentioned.",
        "Explain what should be done according to the guidelines in this excerpt.",
        "Detail the expert suggestions provided in the document.",
        "What proposals are made to address the healthcare issues discussed?"
    ],
    "Table Explanation": [
        "Explain the data shown in the referenced table or figure.",
        "Summarize the contents of the table described in this text.",
        "What key insights can be drawn from the tabular data mentioned?",
        "Describe the structure and findings of the table referenced here.",
        "Interpret the data breakdown provided for the mentioned table."
    ],
    "Narrative": [
        "Summarize the main narrative of this text.",
        "What is the primary topic discussed in the following excerpt?",
        "Explain the key points raised in this detailed description.",
        "Provide a comprehensive overview of this passage.",
        "What are the important contextual details provided here?"
    ],
    "General Healthcare": [
        "Summarize the healthcare information provided in this text.",
        "What are the key health-related findings in this excerpt?",
        "Explain the general medical context discussed below.",
        "Provide an overview of the health topics mentioned.",
        "What are the main takeaways regarding healthcare from this passage?"
    ]
}
