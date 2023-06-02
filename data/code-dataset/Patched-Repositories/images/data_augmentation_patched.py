Traceback (most recent call last):
  File "/home/saurabh/code-energy-consumption/tool/client/script_patcher.py", line 457, in <module>
    main()
  File "/home/saurabh/code-energy-consumption/tool/client/script_patcher.py", line 71, in main
    transf.visit(tree)
  File "/home/saurabh/miniconda3/envs/tf2/lib/python3.9/ast.py", line 407, in visit
    return visitor(node)
  File "/home/saurabh/miniconda3/envs/tf2/lib/python3.9/ast.py", line 483, in generic_visit
    value = self.visit(value)
  File "/home/saurabh/miniconda3/envs/tf2/lib/python3.9/ast.py", line 407, in visit
    return visitor(node)
  File "/home/saurabh/code-energy-consumption/tool/client/script_patcher.py", line 215, in visit_Assign
    modified_node = self.custom_Call(node.value)
  File "/home/saurabh/code-energy-consumption/tool/client/script_patcher.py", line 362, in custom_Call
    value=ast.Constant(ast.unparse(dummyNode).replace(callvisitor_list[0], requiredObjectsSignature.get(callvisitor_list[0]), 1))),
TypeError: replace() argument 2 must be str, not None
