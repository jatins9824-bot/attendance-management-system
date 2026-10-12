import json
import os
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

FILENAME = "students_attendance.json"

# --- DATABASE OPERATIONS ---

def load_data():
    if not os.path.exists(FILENAME):
        return {"students": []}
    with open(FILENAME, "r", encoding="utf-8-sig") as file:
        return json.load(file)

def save_data(data):
    with open(FILENAME, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)

# --- CORE FUNCTIONS ---

def update_attendance(roll_no: str, date: str, status: str) -> str:
    data = load_data()
    for student in data["students"]:
        if student["roll_no"] == str(roll_no):
            student["attendance"][date] = status.capitalize()
            save_data(data)
            return f"Success: {student['name']} (Roll No {roll_no}) marked as {status.capitalize()} for {date}."
    return f"Error: Student with Roll No {roll_no} not found."

def remove_attendance(roll_no: str, date: str) -> str:
    data = load_data()
    for student in data["students"]:
        if student["roll_no"] == str(roll_no) and date in student["attendance"]:
            del student["attendance"][date]
            save_data(data)
            return f"Success: Removed attendance for {student['name']} (Roll No {roll_no}) on {date}."
    return f"Error: Record not found for Roll No {roll_no} on {date}."

def calculate_percentage(roll_no: str) -> str:
    data = load_data()
    for student in data["students"]:
        if student["roll_no"] == str(roll_no):
            records = list(student["attendance"].values())
            if not records:
                return f"No attendance records found for {student['name']}."
            present_count = records.count("Present")
            total_classes = len(records)
            percentage = (present_count / total_classes) * 100
            return (
                f"Student: {student['name']} (Roll No: {roll_no})\n"
                f"Days Present: {present_count}/{total_classes}\n"
                f"Attendance Percentage: {percentage:.2f}%"
            )
    return f"Error: Student with Roll No {roll_no} not found."

# --- EXPLICIT JSON TOOL DEFINITIONS ---
tools = [
    {
        "type": "function",
        "function": {
            "name": "update_attendance",
            "description": "Marks or updates attendance status for a student on a specific date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "roll_no": {"type": "string", "description": "The roll number of the student, e.g. '101'"},
                    "date": {"type": "string", "description": "The date in YYYY-MM-DD format"},
                    "status": {"type": "string", "description": "Status: 'Present' or 'Absent'"}
                },
                "required": ["roll_no", "date", "status"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "remove_attendance",
            "description": "Removes an attendance record for a student on a specific date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "roll_no": {"type": "string", "description": "The roll number of the student, e.g. '101'"},
                    "date": {"type": "string", "description": "The date in YYYY-MM-DD format to remove"}
                },
                "required": ["roll_no", "date"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_percentage",
            "description": "Calculates the attendance percentage for a specific student.",
            "parameters": {
                "type": "object",
                "properties": {
                    "roll_no": {"type": "string", "description": "The roll number of the student, e.g. '101'"}
                },
                "required": ["roll_no"]
            }
        }
    }
]

tool_map = {
    "update_attendance": update_attendance,
    "remove_attendance": remove_attendance,
    "calculate_percentage": calculate_percentage
}

# --- LOAD MODEL ---
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

print("Loading Hugging Face AI model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto"
)

def execute_user_command(user_prompt: str):
    messages = [
        {
            "role": "system",
            "content": (
                "You are an attendance management assistant. Always call the appropriate tool. "
                "Student Directory: 101=Alice Smith, 102=Bob Jones, 103=Charlie Brown, "
                "104=Diana Prince, 105=Ethan Hunt."
            )
        },
        {"role": "user", "content": user_prompt}
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=256)
    raw_response = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=False)

    # Parse and execute tool calls
    match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", raw_response, re.DOTALL)
    if match:
        try:
            tool_data = json.loads(match.group(1))
            func_name = tool_data.get("name")
            func_args = tool_data.get("arguments", {})
            if func_name in tool_map:
                print(f"[Executing Tool]: {func_name}({func_args})")
                return tool_map[func_name](**func_args)
        except Exception as err:
            return f"Error executing tool: {err}"

    return tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)

if __name__ == "__main__":
    print("\n=== Attendance Management System Ready ===")
    print("Type 'exit' to quit.\n")
    while True:
        query = input("Ask AI: ").strip()
        if query.lower() in ["exit", "quit"]:
            break
        if not query:
            continue
        result = execute_user_command(query)
        print(f"\nResult:\n{result}\n" + "-" * 40)