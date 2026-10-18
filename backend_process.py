#!/usr/bin/python
import cgi, json
import cgitb

# Enable detailed error messages in the browser for debugging
cgitb.enable()

# 1. Output HTTP Headers (Must be separated from body by an empty line)
print("Content-Type: text/html")
print()  # Crucial: Blank line signals the end of headers

# 2. Read incoming Form/URL query parameters
form = cgi.FieldStorage()
user_name = form.getvalue("name", "Guest")

# 3. Output HTML content body
# print("<!DOCTYPE html>")
# print("<html>")
# print("<head><title>Python CGI Sample</title></head>")
# print("<body>")
# print(f"<h1>Hello, {user_name}!</h1>")
# print("<p>This page was generated dynamically by a Python CGI script.</p>")
# print("</body>")
# print("</html>")
output_dict={"user_name":user_name,"status":"success"}
print(json.dumps(output_dict))


