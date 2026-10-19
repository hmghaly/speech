#!/usr/bin/python
import json
import os
import sys

# Read the request body
content_length = int(os.environ.get("CONTENT_LENGTH", "0") or 0)
body = sys.stdin.read(content_length) if content_length > 0 else ""

# Parse JSON safely
try:
    data = json.loads(body) if body else {}
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object")
except (json.JSONDecodeError, ValueError):
    print("Status: 400 Bad Request")
    print("Content-Type: application/json; charset=utf-8")
    print()
    print(json.dumps({"status": "error", "message": "Invalid JSON object"}))
    sys.exit(0)

# Extract parameters
user_name = data.get("name", "Guest")

output_dict = {
    "user_name": user_name,
    "status": "success",
    "stdin": body
}

# Output HTTP headers, followed by the JSON response
print("Content-Type: application/json; charset=utf-8")
print()
print(json.dumps(output_dict))



# #!/usr/bin/python
# import cgi, json, sys
# import cgitb

# # Enable detailed error messages in the browser for debugging
# cgitb.enable()

# # 1. Output HTTP Headers (Must be separated from body by an empty line)
# print("Content-Type: text/html")
# print()  # Crucial: Blank line signals the end of headers

# # 2. Read incoming Form/URL query parameters
# form = cgi.FieldStorage()
# user_name = form.getvalue("name", "Guest")


# output_dict={"user_name":user_name,"status":"success"}

# stdin_str=sys.stdin.read()
# output_dict["stdin"]=str(stdin_str)

# print(json.dumps(output_dict))


