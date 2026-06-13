import re
from collections import defaultdict
from typing import Union, Tuple, List

from tree_sitter import Language
from tree_sitter import Parser


from tree_sitter import Language, Parser

class IdentifierAnalyzer:
    def __init__(self, lang="cpp"):
        """Initializes the AST parser and sets up reserved keywords for the specified language."""
        if lang == "c":
            from tree_sitter_c import language as ts_c
            self.language = Language(ts_c())
            query_str = """
            [
                (compound_statement)
                (struct_specifier)
                (function_definition)
                (for_statement)
            ] @scope

            [
                (identifier)
                (field_identifier)
            ] @ident
            """
        elif lang == "cpp":
            from tree_sitter_cpp import language as ts_cpp
            self.language = Language(ts_cpp())
            query_str = """
            [
                (compound_statement)
                (class_specifier)
                (namespace_definition)
                (struct_specifier)
                (function_definition)
                (for_statement)
            ] @scope

            [
                (identifier)
                (field_identifier)
            ] @ident
            """
        else:
            raise ValueError("Unsupported language. Choose 'c' or 'cpp'.")

        # 官方 C/C++ 关键字、宏与特殊内置函数组合
        c_keywords = [
            "auto", "break", "case", "char", "const", "continue", "default", "do", "double", "else", "enum", "extern",
            "float", "for", "goto", "if", "inline", "int", "long", "register", "restrict", "return", "short", "signed",
            "sizeof", "static", "struct", "switch", "typedef", "union", "unsigned", "void", "volatile", "while",
            "_Alignas", "_Alignof", "_Atomic", "_Bool", "_Complex", "_Generic", "_Imaginary", "_Noreturn", "_Static_assert",
            "_Thread_local", "__func__"
        ]
        c_macros = [
            "NULL", "_IOFBF", "_IOLBF", "BUFSIZ", "EOF", "FOPEN_MAX", "TMP_MAX", "FILENAME_MAX", "L_tmpnam",
            "SEEK_CUR", "SEEK_END", "SEEK_SET", "EXIT_FAILURE", "EXIT_SUCCESS", "RAND_MAX", "MB_CUR_MAX"
        ]
        c_special_ids = [
            "main", "stdio", "cstdio", "stdio.h", "size_t", "FILE", "fpos_t", "stdin", "stdout", "stderr",
            "remove", "rename", "tmpfile", "tmpnam", "fclose", "fflush", "fopen", "freopen", "setbuf", "setvbuf",
            "fprintf", "fscanf", "printf", "scanf", "snprintf", "sprintf", "sscanf", "vprintf", "vscanf", "vsnprintf",
            "vsprintf", "vsscanf", "fgetc", "fgets", "fputc", "getc", "getchar", "putc", "putchar", "puts", "ungetc",
            "fread", "fwrite", "fgetpos", "fseek", "fsetpos", "ftell", "rewind", "clearerr", "feof", "ferror", "perror",
            "getline", "stdlib", "cstdlib", "stdlib.h", "div_t", "ldiv_t", "lldiv_t", "atof", "atoi", "atol", "atoll",
            "strtod", "strtof", "strtold", "strtol", "strtoll", "strtoul", "strtoull", "rand", "srand", "aligned_alloc",
            "calloc", "malloc", "realloc", "free", "abort", "atexit", "exit", "at_quick_exit", "_Exit", "getenv",
            "quick_exit", "system", "bsearch", "qsort", "abs", "labs", "llabs", "div", "ldiv", "lldiv", "mblen", "mbtowc",
            "wctomb", "mbstowcs", "wcstombs", "string", "cstring", "string.h", "memcpy", "memmove", "memchr", "memcmp",
            "memset", "strcat", "strncat", "strchr", "strrchr", "strcmp", "strncmp", "strcoll", "strcpy", "strncpy",
            "strerror", "strlen", "strspn", "strcspn", "strpbrk" ,"strstr", "strtok", "strxfrm", "memccpy", "mempcpy",
            "strcat_s", "strcpy_s", "strdup", "strerror_r", "strlcat", "strlcpy", "strsignal", "strtok_r", "iostream",
            "istream", "ostream", "fstream", "sstream", "iomanip", "iosfwd", "ios", "wios", "streamoff", "streampos",
            "wstreampos", "streamsize", "cout", "cerr", "clog", "cin", "boolalpha", "noboolalpha", "skipws", "noskipws",
            "showbase", "noshowbase", "showpoint", "noshowpoint", "showpos", "noshowpos", "unitbuf", "nounitbuf",
            "uppercase", "nouppercase", "left", "right", "internal", "dec", "oct", "hex", "fixed", "scientific", "hexfloat",
            "defaultfloat", "width", "fill", "precision", "endl", "ends", "flush", "ws", "sin", "cos", "tan", "asin",
            "acos", "atan", "atan2", "sinh", "cosh", "tanh", "exp", "sqrt", "log", "log10", "pow", "powf", "ceil", "floor",
            "fabs", "cabs", "frexp", "ldexp", "modf", "fmod", "hypot", "poly", "matherr"
        ]

        # 集合去重
        self.keywords = set(c_keywords + c_macros + c_special_ids)
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
                if parent.type == 'assignment_expression' and parent.child_by_field_name('left') == node:
                    is_write = True
                elif parent.type == 'init_declarator' and parent.child_by_field_name('declarator') == node:
                    is_write = True
                elif parent.type in ('parameter_declaration', 'optional_parameter_declaration'):
                    is_write = True
                elif parent.type == 'update_expression' and parent.child_by_field_name('argument') == node:
                    is_write = True

            dfg_nodes.append(name)

            # =========================================================
            # 🌟 核心修改：将 Byte Offset 转换为 Char Offset，用于对接 HF
            # =========================================================
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
            "type": "global",
            "name": "global"
        }]
        scope_counter = 0

        excluded_parents = {
            'preproc_def',
            'preproc_function_def',
            'type_identifier',
            'template_type',
            'namespace_identifier'
        }

        forbidden_node_types = {
            'destructor_name',
            'operator_name'
        }

        # 1. 核心提速：使用 C 引擎一次性找出所有相关的 scope 和 标识符节点
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
                if node.type in ['class_specifier', 'struct_specifier', 'namespace_definition']:
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
                # 字节级短路验证：极大减少不必要的 UTF-8 解码开销
                name_bytes = source_code[node.start_byte:node.end_byte]
                if name_bytes in self.keywords_bytes:
                    continue

                parent_type = node.parent.type if node.parent else None

                if node.parent and node.parent.type in forbidden_node_types:
                    continue
                if parent_type in excluded_parents:
                    continue

                # 确认是目标变量后，再进行解码
                name = name_bytes.decode("utf-8")

                is_def = False
                curr_node = node
                while curr_node:
                    parent = curr_node.parent
                    if not parent:
                        break

                    # 1. 明确的“使用”场景：作为初始化右值、或数组长度参数
                    if parent.type == 'init_declarator' and parent.child_by_field_name('value') == curr_node:
                        break
                    if parent.type == 'array_declarator' and parent.child_by_field_name('size') == curr_node:
                        break

                    # 2. 如果遇到任何表达式或执行语句，说明它是被调用/使用的变量或函数（例如宏替换、外部全局变量运算）
                    if parent.type in {
                        'binary_expression', 'unary_expression', 'update_expression',
                        'assignment_expression', 'call_expression', 'subscript_expression',
                        'conditional_expression', 'initializer_list', 'initializer',
                        'argument_list', 'return_statement', 'expression_statement',
                        'if_statement', 'while_statement', 'do_statement', 'switch_statement',
                        'case_statement', 'parenthesized_expression', 'cast_expression',
                        'comma_expression', 'sizeof_expression', 'type_descriptor'
                    }:
                        break

                    # 3. 成功到达声明/定义层级，确认此标识符在当前片段中有被定义
                    if parent.type in {
                        'declaration', 'parameter_declaration', 'function_definition',
                        'field_declaration', 'catch_declaration', 'optional_parameter_declaration',
                        'struct_specifier', 'class_specifier', 'enum_specifier'
                    }:
                        is_def = True
                        break

                    curr_node = parent

                if is_def:
                    defined_names.add(name)
                # =====================================================================

                is_func_decl = False
                curr = node.parent
                while curr and curr.type in ['qualified_identifier', 'pointer_declarator', 'reference_declarator',
                                             'parenthesized_declarator']:
                    curr = curr.parent
                if curr and curr.type == "function_declarator":
                    is_func_decl = True

                is_constructor = False

                if node.parent and node.parent.type == "qualified_identifier":
                    scope_node = node.parent.child_by_field_name('scope')
                    name_node = node.parent.child_by_field_name('name')
                    if scope_node and name_node and node == name_node:
                        scope_text = source_code[scope_node.start_byte:scope_node.end_byte].decode("utf-8")
                        scope_basename = scope_text.split("::")[-1]
                        if scope_basename == name:
                            is_constructor = True

                if is_func_decl and not is_constructor:
                    for scope in reversed(scope_stack):
                        if scope["type"] in ['class_specifier', 'struct_specifier']:
                            if name == scope["name"]:
                                is_constructor = True
                            break

                is_method_call = (
                        node.type == "field_identifier" and
                        node.parent and node.parent.type == "field_expression" and
                        node.parent.parent and node.parent.parent.type == "call_expression"
                )

                is_func_call = (
                        parent_type == "call_expression" and
                        node.parent.child_by_field_name('function') == node
                )

                entity_type = "function" if (is_func_decl or is_func_call or is_method_call) else "variable"

                is_plain_field = (node.type == "field_identifier" and not is_method_call and not is_func_decl)

                # =====================================================================
                # [新增逻辑] 提取函数返回值类型 / 变量声明类型
                # =====================================================================
                extracted_type = None
                # 只有在它是定义/声明节点时，才能在 AST 中稳定找到类型
                if is_def or is_func_decl:
                    decl_node = node.parent
                    while decl_node and decl_node.type not in {
                        'function_definition', 'declaration', 'field_declaration',
                        'parameter_declaration', 'optional_parameter_declaration'
                    }:
                        decl_node = decl_node.parent

                    if decl_node:
                        type_node = decl_node.child_by_field_name('type')
                        if type_node:
                            extracted_type = source_code[type_node.start_byte:type_node.end_byte].decode("utf-8")
                # =====================================================================

                if not is_constructor and not is_plain_field:
                    current_scope = scope_stack[-1]
                    identifiers[name].append({
                        "start": node.start_byte,
                        "end": node.end_byte,
                        "scope": current_scope["id"],
                        "scope_start": current_scope["start"],
                        "scope_end": current_scope["end"],
                        "entity_type": entity_type,
                        "return_type": extracted_type  # 统一存入 return_type 字段
                    })

        filtered_identifiers = {
            name: usages
            for name, usages in identifiers.items()
            if name in defined_names
        }

        return filtered_identifiers

    def get_identifier_scope_ranges(self, source_code: bytes, var_name: str):
        """Returns a list of start and end byte ranges defining the scope of a given identifier."""
        identifiers = self.extract_identifiers(source_code)
        if var_name not in identifiers:
            return []

        ranges = set()
        for pos in identifiers[var_name]:
            ranges.add((pos["scope_start"], pos["scope_end"]))
        return list(ranges)

    @staticmethod
    def scopes_overlap(scope_a, scope_b) -> bool:
        """Checks if two distinct scope ranges overlap with one another."""
        a_start, a_end = scope_a
        b_start, b_end = scope_b
        return not (a_end <= b_start or b_end <= a_start)

    def can_rename_to(self, source_code: bytes, old_name: str, new_name: str) -> bool:
        """Determines if an identifier can be safely renamed without causing scope collisions."""
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

    def analyze_format(self, name: str) -> dict:
        """Analyzes an identifier to determine its naming convention and structure."""
        prefix_match = re.match(r'^_+', name)
        prefix = prefix_match.group(0) if prefix_match else ""
        pure_name = name[len(prefix):]

        if not pure_name:
            return {"prefix": prefix, "lengths": [], "style": "special", "count": 0}

        if '_' in pure_name:
            style = "snake_case"
            words = pure_name.split('_')
        else:
            words = re.findall(r'[A-Z]?[a-z0-9]+|[A-Z]+(?=[A-Z][a-z0-9]|\b)', pure_name)
            if pure_name[0].isupper():
                style = "PascalCase"
            else:
                style = "camelCase"

        return {
            "prefix": prefix,
            "lengths": [len(w) for w in words],
            "style": style,
            "count": len(words)
        }

    def canonicalize(self, source_code: Union[str, bytes]) -> str:
        """
        [防御专属] 代码规范化兜底方法：
        将所有提取出的自定义变量和函数替换为泛化 Token (VARx, FUNCx)。
        用于在高方差告警时，物理隔离标识符级别的对抗攻击。
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
        renaming_map = {}

        for name in sorted(identifiers.keys()):
            entity_info = identifiers[name][0]
            entity_type = entity_info.get("entity_type", "variable")

            if entity_type == "function":
                while f"FUNC{func_counter}" in identifiers:
                    func_counter += 1
                renaming_map[name] = f"FUNC{func_counter}"
                func_counter += 1
            else:
                while f"VAR{var_counter}" in identifiers:
                    var_counter += 1
                renaming_map[name] = f"VAR{var_counter}"
                var_counter += 1

        try:
            canonical_code = CodeTransformer.validate_and_apply(
                code_bytes, identifiers, renaming_map, analyzer=None
            )
            return canonical_code
        except Exception as e:
            return code_bytes.decode("utf-8")

    def _get_enclosing_statement(self, node):
        """向上遍历 AST，寻找包含当前节点的最小完整语句块"""
        curr = node
        stop_parent_types = {
            'compound_statement', 'translation_unit', 'for_statement',
            'while_statement', 'do_statement', 'if_statement',
            'switch_statement', 'function_definition'
        }

        while curr.parent:
            if curr.parent.type in stop_parent_types:
                break
            curr = curr.parent
        return curr

    def get_folded_code(self, source_code: bytes, target_var: str) -> str:
        """提取数据流切片，强制闭合分支，向下提取控制流，并保留代表性语句"""
        from tree_sitter import Parser
        parser = Parser()
        parser.language = self.language
        tree = parser.parse(source_code)

        target_nodes = []

        def find_nodes(node):
            if node.type in ["identifier", "field_identifier"]:
                name = source_code[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
                if name == target_var:
                    target_nodes.append(node)
            # 注意：这里已经去掉了无差别的 call_expression 收集，专心跟踪变量
            for child in node.children:
                find_nodes(child)

        find_nodes(tree.root_node)
        if not target_nodes:
            return source_code.decode("utf-8", errors="replace")

        full_nodes = {}

        # 1. 识别并收集核心目标语句
        for node in target_nodes:
            stmt = self._get_enclosing_statement(node)
            if stmt and stmt.id not in full_nodes:
                full_nodes[stmt.id] = stmt

        # 2. 提取完整定义 (Variable Hoisting)
        func_defs = {}
        for stmt in full_nodes.values():
            curr = stmt
            while curr and curr.type != 'function_definition':
                curr = curr.parent
            if curr and curr.id not in func_defs:
                func_defs[curr.id] = curr

        for func in func_defs.values():
            body = func.child_by_field_name('body')
            if body and body.type == 'compound_statement':
                for child in body.children:
                    if child.type == 'declaration':
                        full_nodes[child.id] = child

        # 3.1 标记所有核心节点的祖先路径（骨架 Skeleton）
        skeleton_nodes = {}
        for stmt in full_nodes.values():
            curr = stmt.parent
            while curr:
                skeleton_nodes[curr.id] = curr
                curr = curr.parent

        # 3.2 语义扩展：向下的控制流传播 (Downward Control Flow Propagation)
        CFG_SKELETON_TYPES = {
            'if_statement', 'for_statement', 'while_statement', 'do_statement', 'switch_statement',
            'compound_statement'
        }
        CFG_TERMINAL_TYPES = {
            'goto_statement', 'return_statement', 'break_statement', 'continue_statement'
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
            if node.type in ['if_statement', 'while_statement', 'do_statement', 'switch_statement']:
                cond = node.child_by_field_name('condition')
                if contains_full_node(cond):
                    if node.type == 'if_statement':
                        propagate_control_flow(node.child_by_field_name('consequence'))
                        propagate_control_flow(node.child_by_field_name('alternative'))
                    elif node.type == 'do_statement':
                        propagate_control_flow(node.child_by_field_name('body'))
                    else:
                        propagate_control_flow(node.child_by_field_name('body'))
            elif node.type == 'for_statement':
                init = node.child_by_field_name('initializer')
                cond = node.child_by_field_name('condition')
                upd = node.child_by_field_name('update')
                if contains_full_node(init) or contains_full_node(cond) or contains_full_node(upd):
                    propagate_control_flow(node.child_by_field_name('body'))

        # 3.3 语法修补：单行控制流的边界防吞咽
        def fix_control_flow(node):
            if node.type in ['if_statement', 'for_statement', 'while_statement', 'do_statement']:
                body = node.child_by_field_name(
                    'consequence') if node.type == 'if_statement' else node.child_by_field_name('body')
                if body:
                    if body.type != 'compound_statement':
                        if body.id not in full_nodes:
                            full_nodes[body.id] = body
                    else:
                        if body.id not in skeleton_nodes:
                            skeleton_nodes[body.id] = body
                            fix_control_flow(body)

                if node.type == 'if_statement':
                    alt = node.child_by_field_name('alternative')
                    if alt:
                        if alt.id not in skeleton_nodes:
                            skeleton_nodes[alt.id] = alt
                        for child in alt.children:
                            if child.is_named:
                                if child.type not in ['compound_statement', 'if_statement']:
                                    if child.id not in full_nodes:
                                        full_nodes[child.id] = child
                                else:
                                    if child.id not in skeleton_nodes:
                                        skeleton_nodes[child.id] = child
                                        fix_control_flow(child)

        for node in list(skeleton_nodes.values()):
            fix_control_flow(node)

        # 3.4 块级代表语句补全 (Ensure Block Representativeness)
        # 找出所有被放入骨架的大括号块，按尺寸从内到外(短到长)排序，保证自底向上处理
        comp_stmts = [node for node in skeleton_nodes.values() if node.type == 'compound_statement']
        comp_stmts.sort(key=lambda n: n.end_byte - n.start_byte)

        def has_full_node_inside(n):
            if not n: return False
            if n.id in full_nodes: return True
            for c in n.children:
                if has_full_node_inside(c): return True
            return False

        def contains_call(n):
            if n.type == 'call_expression': return True
            for c in n.children:
                if contains_call(c): return True
            return False

        for comp_node in comp_stmts:
            # 如果这个大括号块里面完全没有保留任何实体语句
            if not has_full_node_inside(comp_node):
                valid_stmts = [c for c in comp_node.children if c.is_named and c.type not in ('comment', 'ERROR')]
                if valid_stmts:
                    picked = None

                    # 优先级 1：寻找包含函数调用的语句（最具代表性）
                    for stmt in valid_stmts:
                        if contains_call(stmt):
                            picked = stmt
                            break

                    # 优先级 2：寻找基础语句，而非复杂的嵌套结构（避免整个嵌套树被不慎完全展开）
                    if not picked:
                        for stmt in valid_stmts:
                            if stmt.type not in ['if_statement', 'for_statement', 'while_statement', 'do_statement',
                                                 'switch_statement']:
                                picked = stmt
                                break

                    # 优先级 3：如果没有别的选择，直接保留第一条可见语句
                    if not picked:
                        picked = valid_stmts[0]

                    # 将选中的这条语句提升为核心保留节点
                    full_nodes[picked.id] = picked

        # 4. 递归遍历 AST，生成保留区间 (Visitor Pattern)
        skeleton_ids = set(skeleton_nodes.keys())
        ranges_to_keep = []

        def traverse(node):
            if node.id in full_nodes:
                ranges_to_keep.append((node.start_byte, node.end_byte))
                return

            if node.id not in skeleton_ids:
                return

            if node.type == 'function_definition':
                body = node.child_by_field_name('body')
                if body:
                    ranges_to_keep.append((node.start_byte, body.start_byte))
                else:
                    ranges_to_keep.append((node.start_byte, node.end_byte))
                for child in node.children:
                    traverse(child)

            elif node.type == 'compound_statement':
                ranges_to_keep.append((node.start_byte, node.start_byte + 1))
                ranges_to_keep.append((node.end_byte - 1, node.end_byte))
                for child in node.children:
                    traverse(child)

            elif node.type == 'if_statement':
                cond = node.child_by_field_name('condition')
                if cond:
                    for child in node.children:
                        if child.type == ')' and child.start_byte >= cond.end_byte:
                            ranges_to_keep.append((node.start_byte, child.end_byte))
                            break
                    else:
                        ranges_to_keep.append((node.start_byte, cond.end_byte))
                else:
                    ranges_to_keep.append((node.start_byte, node.start_byte + 2))

                for child in node.children:
                    if child.type == 'else':
                        alt = node.child_by_field_name('alternative')
                        if alt and (alt.id in skeleton_ids or alt.id in full_nodes):
                            ranges_to_keep.append((child.start_byte, child.end_byte))
                    traverse(child)

            elif node.type in ['for_statement', 'while_statement', 'do_statement', 'switch_statement']:
                cond = node.child_by_field_name('condition')
                if cond:
                    for child in node.children:
                        if child.type == ')' and child.start_byte >= cond.end_byte:
                            ranges_to_keep.append((node.start_byte, child.end_byte))
                            break
                    else:
                        ranges_to_keep.append((node.start_byte, cond.end_byte))
                for child in node.children:
                    traverse(child)

            else:
                for child in node.children:
                    traverse(child)

        traverse(tree.root_node)

        # 5. 区间排序与去重合并
        ranges_to_keep.sort(key=lambda x: x[0])
        merged_ranges = []
        for current in ranges_to_keep:
            if not merged_ranges:
                merged_ranges.append(current)
            else:
                last = merged_ranges[-1]
                if current[0] <= last[1] + 15:
                    merged_ranges[-1] = (last[0], max(last[1], current[1]))
                else:
                    merged_ranges.append(current)

        # 6. 重组文本流
        output = bytearray()
        last_end = 0

        for start, end in merged_ranges:
            gap = start - last_end
            if gap > 15:
                if not (output.endswith(b"/* ... */\n") or output.endswith(b"/* ... */")):
                    output.extend(b"\n    /* ... */\n")
            else:
                output.extend(source_code[last_end:start])

            output.extend(source_code[start:end])
            last_end = end

        if len(source_code) - last_end > 15:
            if not (output.endswith(b"/* ... */\n") or output.endswith(b"/* ... */")):
                output.extend(b"\n    /* ... */\n")
        else:
            output.extend(source_code[last_end:len(source_code)])

        return output.decode("utf-8", errors="replace")

def is_valid_identifier(name: str) -> bool:
    """Validates if a string follows standard C/C++ identifier naming rules."""
    pattern = r'^[a-zA-Z_][a-zA-Z0-9_]*$'
    return bool(re.match(pattern, name))


class CodeTransformer:
    @staticmethod
    def validate_and_apply(source_code: bytes, identifiers: dict, renaming_map: dict, analyzer=None) -> str:
        """Validates renaming rules and securely applies the substitutions to the code bytearray."""
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