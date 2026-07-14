
content = open(r'c:\aura\2\build_final_report.py', encoding='utf-8').read()
# Fix bullet style
content = content.replace('style="List Bullet"', 'style="List Paragraph"')
# Fix bullet run - add bullet char  
content = content.replace('run = p.add_run(text)', 'run = p.add_run(u"\\u2022  " + text)')
open(r'c:\aura\2\build_final_report.py', 'w', encoding='utf-8').write(content)
print('Done.')
