import os

EXCLUDED_DIRS = {
    '.venv', '__pycache__', 'node_modules', '.git',
    'dist', 'build', '.idea', '.vscode'
}

EXCLUDED_EXTENSIONS = {
    '.png', '.jpg', '.jpeg', '.gif', '.bmp',
    '.exe', '.dll', '.zip', '.rar', '.7z',
    '.pdf'
}

def is_text_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            f.read(1000)
        return True
    except:
        return False


def collect_project(project_path, output_file):
    with open(output_file, 'w', encoding='utf-8') as out:

        out.write(f"# Project Export: {project_path}\n\n")

        for root, dirs, files in os.walk(project_path):

            dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]

            for file in files:

                print("READING:", file)

                file_path = os.path.join(root, file)

                if any(file.endswith(ext) for ext in EXCLUDED_EXTENSIONS):
                    continue

                if not is_text_file(file_path):
                    continue

                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                except Exception as e:
                    print("ERROR:", e)
                    continue

                out.write("=" * 80 + "\n")
                out.write(f"## FILE: {file_path}\n")
                out.write("=" * 80 + "\n\n")

                out.write("```python\n")
                out.write(content)
                out.write("\n```\n\n")

    print(f"✅ Done! File created: {output_file}")


if __name__ == "__main__":

    project_folder = r"D:\AI-Projects\Semantic-Matching-Engine"

    output_file = "full_project2.md"

    collect_project(project_folder, output_file)

    print(f"📂 Saved at: {os.path.abspath(output_file)}")