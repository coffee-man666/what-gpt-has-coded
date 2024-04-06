def dict_to_markdown_table(input_dict):
    # Define the table header and separator
    markdown_table = "Property Name | Default Value\n"
    markdown_table += "---|---\n"
    
    # Iterate over dictionary items and add them to the table
    for key, value in input_dict.items():
        markdown_table += f"{key} | {value}\n"
    
    return markdown_table

# Example dictionary
example_dict = {
    "timeout": 10,
    "enable_logging": True,
    "log_level": "INFO",
    "max_connections": 5
}

# Convert the example dictionary to a markdown table
markdown_table = dict_to_markdown_table(example_dict)
print(markdown_table)
