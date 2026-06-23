import re
from collections import defaultdict
from typing import Union, Tuple, List

from tree_sitter import Language, Parser

class IdentifierAnalyzer:
    def __init__(self, lang="python"):
        """Initializes the AST parser and sets up reserved keywords for Python."""
        if lang == "python":
            import tree_sitter_python as ts_python
            # tree-sitter 0.22+ 版本 API
            self.language = Language(ts_python.language())

            # Python 的作用域通常是 module, class, function，为了控制流也加入 block 和语句
            query_str = """
            [
                (module)
                (class_definition)
                (function_definition)
                (for_statement)
                (while_statement)
                (with_statement)
                (if_statement)
            ] @scope

            [
                (identifier)
            ] @ident
            """
        else:
            raise ValueError("This adapted version only supports 'python'.")

        # Python 官方关键字 (keyword.kwlist)
        python_keywords = [
            "False", "None", "True", "and", "as", "assert", "async", "await",
            "break", "class", "continue", "def", "del", "elif", "else", "except",
            "finally", "for", "from", "global", "if", "import", "in", "is",
            "lambda", "nonlocal", "not", "or", "pass", "raise", "return", "try",
            "while", "with", "yield", "match", "case"
        ]

        # Python 常见内置函数、特殊方法与常用标准库对象 (__builtins__)
        python_builtins_and_specials = [
            "abs", "all", "any", "ascii", "bin", "bool", "breakpoint", "bytearray",
            "bytes", "callable", "chr", "classmethod", "compile", "complex", "delattr",
            "dict", "dir", "divmod", "enumerate", "eval", "exec", "filter", "float",
            "format", "frozenset", "getattr", "globals", "hasattr", "hash", "help",
            "hex", "id", "input", "int", "isinstance", "issubclass", "iter", "len",
            "list", "locals", "map", "max", "memoryview", "min", "next", "object",
            "oct", "open", "ord", "pow", "print", "property", "range", "repr",
            "reversed", "round", "set", "setattr", "slice", "sorted", "staticmethod",
            "str", "sum", "super", "tuple", "type", "vars", "zip", "__import__",
            "__init__", "__main__", "__name__", "self", "cls", "Exception", "ValueError",
            "TypeError", "IndexError", "KeyError", "sys", "os", "math", "re"
        ]

        # 集合去重
        self.keywords = set(python_keywords + python_builtins_and_specials)
        self.keywords_bytes = {k.encode("utf-8") for k in self.keywords}

        try:
            from tree_sitter import Query
            self.ast_query = Query(self.language, query_str)
        except ImportError:
            self.ast_query = self.language.query(query_str)

    def extract_dataflow(self, source_code: bytes) -> Tuple[List[str], List[Tuple[int, int]], List[List[int]]]:
        parser = Parser()
        parser.language = self.language
        tree = parser.parse(source_code)

        if hasattr(self.ast_query, "captures"):
            captures = self.ast_query.captures(tree.root_node)
        else:
            from tree_sitter import QueryCursor
            cursor = QueryCursor(self.ast_query)
            captures = cursor.captures(tree.root_node)

        if isinstance(captures, dict):
            flat_captures = [(n, t) for t, nodes in captures.items() for n in
                             (nodes if isinstance(nodes, list) else [nodes])]
        else:
            flat_captures = list(captures)

        flat_captures.sort(key=lambda x: (x[0].start_byte, -x[0].end_byte))

        dfg_nodes = []
        dfg_to_code = []
        dfg_to_dfg = []
        last_write_state = defaultdict(list)
        current_node_idx = 0

        for node, tag in flat_captures:
            if tag != "ident":
                continue

            name_bytes = source_code[node.start_byte:node.end_byte]
            if name_bytes in self.keywords_bytes:
                continue

            name = name_bytes.decode("utf-8", errors="replace")

            is_write = False
            parent = node.parent
            if parent:
                # Python 赋值和数据流特征节点
                if parent.type in ('assignment', 'annassignment', 'augmented_assignment'):
                    left_node = parent.child_by_field_name('left') or parent.child_by_field_name('pattern')
                    if left_node == node or (
                            left_node and left_node.type == 'pattern_list' and node in left_node.children):
                        is_write = True
                elif parent.type in ('parameters', 'default_parameter', 'typed_parameter', 'typed_default_parameter'):
                    is_write = True
                elif parent.type == 'for_statement' and parent.child_by_field_name('left') == node:
                    is_write = True
                elif parent.type == 'with_item' and parent.child_by_field_name('value') == node:  # as xxx
                    is_write = True

            dfg_nodes.append(name)

            # 将 Byte Offset 转换为 Char Offset，用于对接模型 Tokenizer
            start_char = len(source_code[:node.start_byte].decode("utf-8", errors="replace"))
            end_char = len(source_code[:node.end_byte].decode("utf-8", errors="replace"))
            dfg_to_code.append((start_char, end_char))

            incoming_edges = []
            if not is_write:
                if name in last_write_state:
                    incoming_edges.extend(last_write_state[name])

            incoming_edges.append(current_node_idx)  # 自环
            dfg_to_dfg.append(list(set(incoming_edges)))

            if is_write:
                last_write_state[name] = [current_node_idx]

            current_node_idx += 1

        return dfg_nodes, dfg_to_code, dfg_to_dfg

    def extract_identifiers(self, source_code: bytes) -> dict:
        """Traverses the AST to extract non-keyword identifiers and their scope information."""
        parser = Parser()
        parser.language = self.language
        tree = parser.parse(source_code)
        identifiers = defaultdict(list)

        defined_names = set()

        scope_stack = [{
            "id": 0,
            "start": 0,
            "end": len(source_code),
            "type": "module",
            "name": "global"
        }]
        scope_counter = 0

        # 在 Python 中，我们不提取模块导入产生的别名或内置字段
        excluded_parents = {
            'import_statement',
            'import_from_statement',
            'import_prefix',
            'aliased_import',
            'string'
        }

        if hasattr(self.ast_query, "captures"):
            captures = self.ast_query.captures(tree.root_node)
        else:
            from tree_sitter import QueryCursor
            cursor = QueryCursor(self.ast_query)
            captures = cursor.captures(tree.root_node)

        if isinstance(captures, dict):
            flat_captures = [(node, tag) for tag, nodes in captures.items() for node in
                             (nodes if isinstance(nodes, list) else [nodes])]
        else:
            flat_captures = list(captures)

        flat_captures.sort(key=lambda x: (x[0].start_byte, -x[0].end_byte))

        for node, tag in flat_captures:

            while len(scope_stack) > 1 and scope_stack[-1]["end"] <= node.start_byte:
                scope_stack.pop()

            if tag == "scope":
                scope_counter += 1
                scope_name = ""
                if node.type in ['class_definition', 'function_definition']:
                    name_node = node.child_by_field_name('name')
                    if name_node:
                        scope_name = source_code[name_node.start_byte:name_node.end_byte].decode("utf-8")

                scope_stack.append({
                    "id": scope_counter,
                    "start": node.start_byte,
                    "end": node.end_byte,
                    "type": node.type,
                    "name": scope_name
                })

            elif tag == "ident":
                name_bytes = source_code[node.start_byte:node.end_byte]
                if name_bytes in self.keywords_bytes:
                    continue

                parent_type = node.parent.type if node.parent else None

                if parent_type in excluded_parents:
                    continue

                name = name_bytes.decode("utf-8")

                is_def = False
                curr_node = node
                while curr_node:
                    parent = curr_node.parent
                    if not parent:
                        break

                    # Python 的变量定义往往出现在左值、参数、类名或函数名
                    if parent.type in {
                        'assignment', 'annassignment', 'augmented_assignment',
                        'function_definition', 'class_definition', 'parameters',
                        'for_statement', 'with_item', 'except_clause', 'pattern_list'
                    }:
                        is_def = True
                        break

                    # 如果遇到任何运算、调用、表达式，停止向外判断定义
                    if parent.type in {
                        'binary_operator', 'unary_operator', 'boolean_operator', 'comparison_operator',
                        'call', 'subscript', 'attribute', 'list', 'dictionary', 'set', 'tuple',
                        'return_statement', 'if_statement', 'while_statement', 'expression_statement',
                        'yield', 'await'
                    }:
                        break

                    curr_node = parent

                if is_def:
                    defined_names.add(name)

                # 判断类型：是变量、函数、类还是属性
                is_func_def = (parent_type == "function_definition" and node.parent.child_by_field_name('name') == node)
                is_class_def = (parent_type == "class_definition" and node.parent.child_by_field_name('name') == node)

                is_func_call = (parent_type == "call" and node.parent.child_by_field_name('function') == node)
                is_attribute = (parent_type == "attribute" and node.parent.child_by_field_name('attribute') == node)

                if is_func_def or is_func_call:
                    entity_type = "function"
                elif is_class_def:
                    entity_type = "class"
                elif is_attribute:
                    entity_type = "attribute"
                else:
                    entity_type = "variable"

                extracted_type = None
                if is_def and parent_type in ('annassignment', 'typed_parameter', 'function_definition'):
                    type_node = node.parent.child_by_field_name('type') or node.parent.child_by_field_name(
                        'return_type')
                    if type_node:
                        extracted_type = source_code[type_node.start_byte:type_node.end_byte].decode("utf-8")

                current_scope = scope_stack[-1]
                identifiers[name].append({
                    "start": node.start_byte,
                    "end": node.end_byte,
                    "scope": current_scope["id"],
                    "scope_start": current_scope["start"],
                    "scope_end": current_scope["end"],
                    "entity_type": entity_type,
                    "return_type": extracted_type
                })

        # 过滤掉那些使用过，但未在当前代码中定义的外部变量/模块
        filtered_identifiers = {
            name: usages
            for name, usages in identifiers.items()
            if name in defined_names
        }

        return filtered_identifiers

    def get_identifier_scope_ranges(self, source_code: bytes, var_name: str):
        identifiers = self.extract_identifiers(source_code)
        if var_name not in identifiers:
            return []

        ranges = set()
        for pos in identifiers[var_name]:
            ranges.add((pos["scope_start"], pos["scope_end"]))
        return list(ranges)

    @staticmethod
    def scopes_overlap(scope_a, scope_b) -> bool:
        a_start, a_end = scope_a
        b_start, b_end = scope_b
        return not (a_end <= b_start or b_end <= a_start)

    def can_rename_to(self, source_code: bytes, old_name: str, new_name: str) -> bool:
        identifiers = self.extract_identifiers(source_code)
        if old_name == new_name:
            return False
        if new_name not in identifiers:
            return True
        if old_name not in identifiers:
            return False

        old_scopes = {(p["scope_start"], p["scope_end"]) for p in identifiers[old_name]}
        new_scopes = {(p["scope_start"], p["scope_end"]) for p in identifiers[new_name]}

        for oscope in old_scopes:
            for nscope in new_scopes:
                if self.scopes_overlap(oscope, nscope):
                    return False
        return True

    def canonicalize(self, source_code: Union[str, bytes]) -> str:
        """
        Authorship Attribution 特别适用：抹除开发者自定义变量名的痕迹
        强迫分类模型学习结构化特征，而非死记变量名。
        """
        if isinstance(source_code, str):
            code_bytes = source_code.encode("utf-8")
        else:
            code_bytes = source_code

        identifiers = self.extract_identifiers(code_bytes)
        if not identifiers:
            return code_bytes.decode("utf-8")

        var_counter = 1
        func_counter = 1
        class_counter = 1
        renaming_map = {}

        for name in sorted(identifiers.keys()):
            entity_info = identifiers[name][0]
            entity_type = entity_info.get("entity_type", "variable")

            if entity_type == "function":
                while f"FUNC_{func_counter}" in identifiers:
                    func_counter += 1
                renaming_map[name] = f"FUNC_{func_counter}"
                func_counter += 1
            elif entity_type == "class":
                while f"CLASS_{class_counter}" in identifiers:
                    class_counter += 1
                renaming_map[name] = f"CLASS_{class_counter}"
                class_counter += 1
            elif entity_type != "attribute":  # Python属性通常不随便替换以防破坏库调用
                while f"VAR_{var_counter}" in identifiers:
                    var_counter += 1
                renaming_map[name] = f"VAR_{var_counter}"
                var_counter += 1

        try:
            canonical_code = CodeTransformer.validate_and_apply(
                code_bytes, identifiers, renaming_map, analyzer=None
            )
            return canonical_code
        except Exception as e:
            return code_bytes.decode("utf-8")

    def _get_enclosing_statement(self, node):
        curr = node
        stop_parent_types = {
            'module', 'function_definition', 'class_definition', 'for_statement',
            'while_statement', 'if_statement', 'with_statement', 'try_statement',
            'match_statement', 'expression_statement'
        }

        while curr.parent:
            if curr.parent.type in stop_parent_types:
                break
            curr = curr.parent
        return curr

    def get_folded_code(self, source_code: bytes, target_var: str) -> str:
        parser = Parser()
        parser.language = self.language
        tree = parser.parse(source_code)

        target_nodes = []

        def find_nodes(node):
            if node.type == "identifier":
                name = source_code[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
                if name == target_var:
                    target_nodes.append(node)
            for child in node.children:
                find_nodes(child)

        find_nodes(tree.root_node)
        if not target_nodes:
            return source_code.decode("utf-8", errors="replace")

        full_nodes = {}

        for node in target_nodes:
            stmt = self._get_enclosing_statement(node)
            if stmt and stmt.id not in full_nodes:
                full_nodes[stmt.id] = stmt

        func_defs = {}
        for stmt in full_nodes.values():
            curr = stmt
            while curr and curr.type != 'function_definition':
                curr = curr.parent
            if curr and curr.id not in func_defs:
                func_defs[curr.id] = curr

        skeleton_nodes = {}
        for stmt in full_nodes.values():
            curr = stmt.parent
            while curr:
                skeleton_nodes[curr.id] = curr
                curr = curr.parent

        CFG_SKELETON_TYPES = {
            'if_statement', 'for_statement', 'while_statement', 'try_statement', 'with_statement',
            'match_statement', 'block'
        }
        CFG_TERMINAL_TYPES = {
            'return_statement', 'break_statement', 'continue_statement', 'raise_statement', 'yield'
        }

        def contains_full_node(n):
            if not n: return False
            if n.id in full_nodes: return True
            for c in n.children:
                if contains_full_node(c): return True
            return False

        def propagate_control_flow(node):
            if not node: return
            is_target = False
            if node.type in CFG_SKELETON_TYPES:
                skeleton_nodes[node.id] = node
                is_target = True
            elif node.type in CFG_TERMINAL_TYPES:
                full_nodes[node.id] = node
                is_target = True

            if is_target:
                curr = node.parent
                while curr and curr.id not in skeleton_nodes:
                    skeleton_nodes[curr.id] = curr
                    curr = curr.parent

            for child in node.children:
                propagate_control_flow(child)

        for node in list(skeleton_nodes.values()):
            if node.type in ['if_statement', 'while_statement', 'match_statement']:
                cond = node.child_by_field_name('condition') or node.child_by_field_name('subject')
                if contains_full_node(cond):
                    if node.type == 'if_statement':
                        propagate_control_flow(node.child_by_field_name('consequence'))
                        propagate_control_flow(node.child_by_field_name('alternative'))
                    else:
                        propagate_control_flow(node.child_by_field_name('body'))
            elif node.type == 'for_statement':
                left = node.child_by_field_name('left')
                right = node.child_by_field_name('right')
                if contains_full_node(left) or contains_full_node(right):
                    propagate_control_flow(node.child_by_field_name('body'))

        comp_stmts = [node for node in skeleton_nodes.values() if node.type == 'block']
        comp_stmts.sort(key=lambda n: n.end_byte - n.start_byte)

        def has_full_node_inside(n):
            if not n: return False
            if n.id in full_nodes: return True
            for c in n.children:
                if has_full_node_inside(c): return True
            return False

        for comp_node in comp_stmts:
            if not has_full_node_inside(comp_node):
                valid_stmts = [c for c in comp_node.children if c.is_named and c.type not in ('comment', 'ERROR')]
                if valid_stmts:
                    picked = valid_stmts[0]
                    full_nodes[picked.id] = picked

        skeleton_ids = set(skeleton_nodes.keys())
        ranges_to_keep = []

        def traverse(node):
            if node.id in full_nodes:
                ranges_to_keep.append((node.start_byte, node.end_byte))
                return

            if node.id not in skeleton_ids:
                return

            if node.type in ['function_definition', 'class_definition']:
                body = node.child_by_field_name('body')
                if body:
                    ranges_to_keep.append((node.start_byte, body.start_byte))
                else:
                    ranges_to_keep.append((node.start_byte, node.end_byte))
                for child in node.children:
                    traverse(child)

            elif node.type == 'block':
                for child in node.children:
                    traverse(child)

            elif node.type in ['if_statement', 'for_statement', 'while_statement', 'with_statement', 'try_statement']:
                # 在 Python 中，保留头部（带冒号的部分）
                body = node.child_by_field_name('body') or node.child_by_field_name('consequence')
                if body:
                    ranges_to_keep.append((node.start_byte, body.start_byte))
                for child in node.children:
                    if child.type == 'elif_clause' or child.type == 'else_clause':
                        ranges_to_keep.append((child.start_byte, child.child_by_field_name(
                            'body').start_byte if child.child_by_field_name('body') else child.end_byte))
                    traverse(child)
            else:
                for child in node.children:
                    traverse(child)

        traverse(tree.root_node)

        ranges_to_keep.sort(key=lambda x: x[0])
        merged_ranges = []
        for current in ranges_to_keep:
            if not merged_ranges:
                merged_ranges.append(current)
            else:
                last = merged_ranges[-1]
                if current[0] <= last[1] + 5:  # Python 空白符容忍度较低，缩短合并距离
                    merged_ranges[-1] = (last[0], max(last[1], current[1]))
                else:
                    merged_ranges.append(current)

        output = bytearray()
        last_end = 0

        for start, end in merged_ranges:
            gap = start - last_end
            if gap > 15:
                if not (output.endswith(b"# ...\n") or output.endswith(b"# ...")):
                    output.extend(b"\n# ...\n")
            else:
                output.extend(source_code[last_end:start])

            output.extend(source_code[start:end])
            last_end = end

        if len(source_code) - last_end > 15:
            if not (output.endswith(b"# ...\n") or output.endswith(b"# ...")):
                output.extend(b"\n# ...\n")
        else:
            output.extend(source_code[last_end:len(source_code)])

        return output.decode("utf-8", errors="replace")


def is_valid_identifier(name: str) -> bool:
    """Validates if a string follows standard Python identifier naming rules."""
    pattern = r'^[a-zA-Z_][a-zA-Z0-9_]*$'
    return bool(re.match(pattern, name))


class CodeTransformer:
    @staticmethod
    def validate_and_apply(source_code: bytes, identifiers: dict, renaming_map: dict, analyzer=None) -> str:
        for old_name, new_name in renaming_map.items():
            if not is_valid_identifier(new_name):
                raise ValueError(f"Invalid naming: '{new_name}'")

            if analyzer is not None:
                if not analyzer.can_rename_to(source_code, old_name, new_name):
                    raise ValueError(f"Scope conflict: '{old_name}' -> '{new_name}' is unavailable.")
            else:
                existing_names = set(identifiers.keys())
                if new_name in existing_names and new_name != old_name:
                    raise ValueError(f"Renaming conflict: '{old_name}' -> '{new_name}' already exists.")

        code = bytearray(source_code)
        replacements = []
        for old_name, new_name in renaming_map.items():
            if old_name in identifiers:
                for pos in identifiers[old_name]:
                    replacements.append((pos['start'], pos['end'], new_name))

        replacements.sort(key=lambda x: x[0], reverse=True)
        for start, end, new_name in replacements:
            code[start:end] = new_name.encode("utf-8")

        return code.decode("utf-8")