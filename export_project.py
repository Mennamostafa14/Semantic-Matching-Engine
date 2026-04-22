import os

# فولدرات مش محتاجينها
EXCLUDED_DIRS = {
    '.venv', '__pycache__', 'node_modules', '.git',
    'dist', 'build', '.idea', '.vscode'
}

# أنواع ملفات مش مفيدة (صور / ملفات تنفيذية)
EXCLUDED_EXTENSIONS = {
    '.png', '.jpg', '.jpeg', '.gif', '.bmp',
    '.exe', '.dll', '.zip', '.rar', '.7z',
    '.pdf'
}

def is_text_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            f.read()
        return True
    except:
        return False


def collect_project(project_path, output_file):
    with open(output_file, 'w', encoding='utf-8') as out:

        out.write(f"# Project Export: {project_path}\n\n")

        for root, dirs, files in os.walk(project_path):
            # استبعاد فولدرات غير مهمة
            dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]

            for file in files:
                file_path = os.path.join(root, file)

                # استبعاد ملفات معينة
                if any(file.endswith(ext) for ext in EXCLUDED_EXTENSIONS):
                    continue

                # قراءة الملفات النصية فقط
                if not is_text_file(file_path):
                    continue

                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                except:
                    continue

                # كتابة في الملف النهائي
                out.write("=" * 80 + "\n")
                out.write(f"## FILE: {file_path}\n")
                out.write("=" * 80 + "\n\n")

                out.write("``` \n")
                out.write(content)
                out.write("\n``` \n\n\n")

    print(f"✅ Done! File created: {output_file}")


if __name__ == "__main__":
    # ✏️ عدلي هنا بس
    project_folder = r"D:\AI-Projects\Semantic-Matching-Engine\src"   # حطي مسار مشروعك
    output_file = "full_project1.md"

    collect_project(project_folder, output_file)
    print(f"📂 Saved at: {os.path.abspath(output_file)}")