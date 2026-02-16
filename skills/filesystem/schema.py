from pydantic import BaseModel, Field

class ReadFileInput(BaseModel):
    file_path: str = Field(..., description="The absolute path to the file to be read. Example: 'C:/Users/macht/SA/sgr_core/server.py'")
    max_lines: int = Field(2000, description="Maximum number of lines to read. Default is 2000.")

class ListDirInput(BaseModel):
    dir_path: str = Field(..., description="The absolute path to the directory to list. Example: 'C:/Users/macht/SA'")
