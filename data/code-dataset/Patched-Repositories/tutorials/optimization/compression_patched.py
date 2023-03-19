Traceback (most recent call last):
  File "/home/saurabh/code-energy-consumption/tool/client/script_patcher.py", line 308, in <module>
    main()
  File "/home/saurabh/code-energy-consumption/tool/client/script_patcher.py", line 28, in main
    tree = ast.parse(sourceCode)
  File "/home/saurabh/miniconda3/envs/tf2/lib/python3.9/ast.py", line 50, in parse
    return compile(source, filename, mode, flags,
  File "<unknown>", line 2
    read MAJOR MINOR <<< "$(pip show tensorflow | perl -p -0777 -e 's/.*Version: (\d+)\.(\d+).*/\1 \2/sg')"
         ^
SyntaxError: invalid syntax
