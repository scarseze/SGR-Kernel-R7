from typing import List, Optional, Literal
from pydantic import BaseModel, Field

class FunctionReq(BaseModel):
    name: str = Field(description="Name of the subsystem or function")
    description: str = Field(description="Description of what it does")

class GostData(BaseModel):
    """
    Data structure for GOST 34 Technical Specification (TZ).
    """
    system_name: str = Field(..., description="Short name of the system")
    system_full_name: str = Field(..., description="Full official name")
    contract_number: str = Field(default="№ 01/2026", description="Contract number")
    customer_name: str = Field(description="Name of the customer organization")
    developer_name: str = Field(description="Name of the developer organization")
    foundation_documents: str = Field(default="Договор № 01/2026 от 01.01.2026", description="Documents serving as basis")
    start_date: str = Field(default="01.02.2026")
    end_date: str = Field(default="31.12.2026")
    financing_source: str = Field(default="Собственные средства Заказчика")
    delivery_procedure: str = Field(default="По Акту сдачи-приемки работ")
    
    purpose: str = Field(description="Main purpose of the system")
    goals: List[str] = Field(description="List of specific goals")
    object_description: str = Field(description="Description of the automation object")
    conditions: str = Field(description="Operating conditions")
    
    requirements_structure: str = Field(description="Requirements for structure and functioning")
    requirements_personnel: str = Field(default="Персонал должен иметь навыки работы с ПК.")
    requirements_indicators: str = Field(description="Key performance indicators")
    requirements_reliability: str = Field(description="Reliability requirements")
    requirements_safety: str = Field(description="Safety requirements")
    requirements_ergonomics: str = Field(default="Интерфейс должен быть интуитивно понятным.")
    requirements_transport: str = Field(default="Не предъявляются.")
    requirements_maintenance: str = Field(description="Maintenance requirements")
    requirements_security: str = Field(description="Information security requirements")
    requirements_data_protection: str = Field(description="Data protection requirements")
    requirements_environment: str = Field(default="Офисное помещение.")
    requirements_patent: str = Field(default="Патентная чистота должна быть обеспечена.")
    requirements_standardization: str = Field(default="Система должна соответствовать ГОСТ 34.")
    
    functions: List[FunctionReq] = Field(description="List of system functions/subsystems")
    
    support_math: str = Field(default="Стандартные алгоритмы.", description="Mathematical support")
    support_info: str = Field(description="Information support (DB, formats)")
    support_linguo: str = Field(default="Русский язык интерфейса.")
    support_software: str = Field(description="Software requirements (OS, DBMS)")
    support_hardware: str = Field(description="Hardware requirements")
    support_metrology: str = Field(default="Не предъявляются.")
    support_org: str = Field(description="Organizational support")
    support_method: str = Field(default="Руководство пользователя.")
    
    works_content: str = Field(description="List of stages and works")
    testing_types: str = Field(description="Types of testing (PSI, PMI)")
    acceptance_requirements: str = Field(default="В соответствии с ПМИ.")
    preparation_works: str = Field(description="Works to prepare the facility")
    documentation_requirements: str = Field(description="List of documentation to be delivered")
    development_sources: str = Field(default="ТЗ, ГОСТ 34.602-89")

class GostWriterInput(BaseModel):
    """
    Input for the GostWriter Skill.
    """
    action: Literal["generate"] = Field(
        ..., 
        description="Action to perform. 'generate' creates a new document content."
    )
    topic: str = Field(
        ..., 
        description="Brief description of the system to write documentation for (e.g. 'CRM for Bank')."
    )
    template_type: Literal["tz", "manual", "pmi"] = Field(
        default="tz",
        description="Type of document to generate."
    )
