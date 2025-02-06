
from executor import BRDRAG

brdrag = BRDRAG()
response_DDA = brdrag.getResponse(["MATERIAL MANAGEMENT - KDS UPDATED.xlsx"], "What are different Business Priorities?")
print("Response for DDA: ")
print(response_DDA)

#response_MOM = brdrag.getResponse(["2023.12.18_Tamkeen_Feed_Source to Pay MoM_V1.0.docx"], "What are the action items?")
#print("Response for MOM: ")
#print(response_MOM)


#response_KDS = brdrag.getResponse(["MATERIAL MANAGEMENT - KDS UPDATED.xlsx"], "What are different KDS List?")
#print("Response for KDS: ")
#print(response_DDA)

    