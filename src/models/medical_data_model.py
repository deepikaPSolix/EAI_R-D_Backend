from pydantic import BaseModel, Field
from typing import List


class ExtractedData(BaseModel):
    full_name: str = Field(default="", description="Full Name of the patient")
    address_info: str = Field(default="", description="Address information of the patient")
    email: str = Field(default="", description="Email address of the patient")
    phone_numbers: List[str] = Field(default_factory=list, description="List of phone numbers")
    ssn: str = Field(default="", description="Social Security Number")
    drivers_license: str = Field(default="", description="Driver's License Number")
    passport_number: str = Field(default="", description="Passport Number")
    national_id: str = Field(default="", description="National Identification Number")
    bank_accounts: List[str] = Field(default_factory=list, description="List of Bank Account Numbers")
    credit_cards: List[str] = Field(default_factory=list, description="List of Credit Card Numbers")
    debit_cards: List[str] = Field(default_factory=list, description="List of Debit Card Numbers")
    payment_card_details: str = Field(default="", description="Details of the payment card")
    medical_records: bool = Field(default=False, description="Is this a medical document?")
    health_insurance_info: bool = Field(default=False, description="Does it have health insurance information?")
    patient_identifiers: str = Field(default="", description="Patient identifiers in medical records")
    fingerprints: List[str] = Field(default_factory=list, description="Biometric data used for identification based on finger patterns.")
    facial_recognition_data: List[str] = Field(default_factory=list, description="Biometric data used for identification based on facial features.")
    voiceprints: List[str] = Field(default_factory=list, description="Biometric data used for identification based on voice patterns.")
    retina_iris_scans: List[str] = Field(default_factory=list, description="Biometric data used for identification based on eye patterns.")
    date_of_birth: str = Field(default_factory="", description="The date on which a person was born.")
    place_of_birth: str = Field(default_factory="", description="The location where a person was born.")
    company_name:str = Field(default= "", description="The name of the company or organization that the document belongs to.")