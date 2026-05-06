import sys
import re
import random

# --- 基础词法分析与 AST 构建 ---

token_pat = re.compile(r"""
    (?P<COMMENT_ML>/\*[\s\S]*?\*/) |
    (?P<COMMENT_SL>//[^\n]*(?:\n|$)) |
    (?P<RAW_STRING>R"(?P<delim>[^ ()\\\t\v\n]*)\([\s\S]*?\)\(?P=delim\)") |
    (?P<STRING>L?"(?:\\.|[^"\\\n])*") |
    (?P<CHAR>'(?:\\.|[^'\\\n])*') |
    (?P<PREPROC>^[ \t]*\#(?:[^\\\n]|\\.)*(?:\n|$)) |
    (?P<OP_ASSIGN><<=|>>=|\+=|-=|\*=|/=|%=|&=|\|=|\^=|=(?!=)) |
    (?P<OP>==|!=|<=|>=|<<|>>|&&|\|\||\+\+|--|->|\+|-|\*|/|%|&|\||\^|!|<|>|\?|:|\.) |
    (?P<ID>[a-zA-Z_]\w*) |
    (?P<NUMBER>(?:0[xX][0-9a-fA-F]+|[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?[uUlLfF]*)) |
    (?P<PUNCT>[(){}\[\];,]) |
    (?P<SPACE>\s+) |
    (?P<MISC>.)
""", re.MULTILINE | re.VERBOSE)

CLOSERS = {'(': ')', '{': '}', '[': ']'}

def build_ast(tokens):
    stack = [[]]
    openers = []
    for kind, val in tokens:
        if kind == 'PUNCT' and val in '({[':
            stack.append([])
            openers.append((kind, val))
        elif kind == 'PUNCT' and val in ')}]':
            if openers:
                op_kind, op_val = openers.pop()
                inner = stack.pop()
                stack[-1].append(('GROUP', op_val + val, inner))
            else:
                stack[-1].append((kind, val))
        else:
            stack[-1].append((kind, val))
    while openers:
        op_kind, op_val = openers.pop()
        inner = stack.pop()
        stack[-1].append(('GROUP', op_val + '?', inner))
    return stack[0]

def flatten_ast(ast_list):
    res = []
    for node in ast_list:
        if node[0] == 'GROUP':
            opener = node[1][0]
            closer = node[1][1] if len(node[1]) > 1 and node[1][1] in ')}]' else CLOSERS.get(opener, '')
            res.append(opener)
            res.extend(flatten_ast(node[2]))
            res.append(closer)
        else:
            res.append(node[1])
    return "".join(res)

# --- Transformation 1: Else Padding (补全 else { }) ---

def apply_else_padding(code):
    if not code: return code
    token_pattern = re.compile(r"""
        (?P<STR>\"(?:\\.|[^\\\"])*\") |
        (?P<CHAR>\'(?:\\.|[^\\\'])*\') |
        (?P<LINE_COM>//[^\n]*) |
        (?P<BLK_COM>/\*.*?\*/) |
        (?P<PREPROC>^[ \t]*\#(?:[^\n]*\\\n)*[^\n]*) |
        (?P<IF>\bif\b) |
        (?P<ELSE>\belse\b) |
        (?P<WHILE>\bwhile\b) |
        (?P<FOR>\bfor\b) |
        (?P<DO>\bdo\b) |
        (?P<LPAREN>\() |
        (?P<RPAREN>\)) |
        (?P<LBRACE>\{) |
        (?P<RBRACE>\}) |
        (?P<SEMI>;) |
        (?P<WS>\s+) |
        (?P<OTHER>[a-zA-Z0-9_]+|[^a-zA-Z0-9_ \t\n\r\"\'/(){};]+|/)
    """, re.DOTALL | re.VERBOSE | re.MULTILINE)
    tokens = []
    for match in token_pattern.finditer(code):
        tokens.append((match.lastgroup, match.group(), match.start(), match.end()))
    insertions = []
    def skip_ignored(i):
        while i < len(tokens) and tokens[i][0] in ('WS', 'LINE_COM', 'BLK_COM', 'PREPROC'): i += 1
        return i
    def parse_parens(i):
        i += 1; depth = 1
        while i < len(tokens) and depth > 0:
            if tokens[i][0] == 'LPAREN': depth += 1
            elif tokens[i][0] == 'RPAREN': depth -= 1
            i += 1
        return i
    def parse_statement(i):
        i = skip_ignored(i)
        if i >= len(tokens): return i
        tok_type = tokens[i][0]
        if tok_type == 'LBRACE':
            i += 1
            while i < len(tokens):
                i = skip_ignored(i)
                if i >= len(tokens) or tokens[i][0] == 'RBRACE': break
                i = parse_statement(i)
            if i < len(tokens) and tokens[i][0] == 'RBRACE': i += 1
            return i
        elif tok_type == 'IF':
            i += 1; i = skip_ignored(i)
            if i < len(tokens) and tokens[i][0] == 'LPAREN': i = parse_parens(i)
            i = parse_statement(i); end_of_if_body = i
            i_next = skip_ignored(i)
            if i_next < len(tokens) and tokens[i_next][0] == 'ELSE':
                i = i_next + 1; i = parse_statement(i)
            else:
                last_tok_idx = end_of_if_body - 1
                if last_tok_idx >= 0: insertions.append((tokens[last_tok_idx][3], " else { }"))
            return i
        elif tok_type in ('WHILE', 'FOR'):
            i += 1; i = skip_ignored(i)
            if i < len(tokens) and tokens[i][0] == 'LPAREN': i = parse_parens(i)
            i = parse_statement(i); return i
        elif tok_type == 'DO':
            i += 1; i = parse_statement(i); i_next = skip_ignored(i)
            if i_next < len(tokens) and tokens[i_next][0] == 'WHILE':
                i = i_next + 1; i = skip_ignored(i)
                if i < len(tokens) and tokens[i][0] == 'LPAREN': i = parse_parens(i)
                i = skip_ignored(i)
                if i < len(tokens) and tokens[i][0] == 'SEMI': i += 1
            return i
        else:
            t = tokens[i][0]; i += 1
            if t == 'SEMI': return i
            if t in ('LBRACE', 'RBRACE', 'IF', 'WHILE', 'FOR', 'DO', 'ELSE'): return i
            while i < len(tokens):
                t = tokens[i][0]
                if t == 'SEMI': i += 1; break
                if t in ('LBRACE', 'RBRACE', 'IF', 'WHILE', 'FOR', 'DO', 'ELSE'): break
                i += 1
            return i
    i = 0
    while i < len(tokens):
        i_prev = i; i = skip_ignored(i)
        if i >= len(tokens): break
        i = parse_statement(i)
        if i == i_prev: i += 1
    insertion_map = {}
    for idx, text in insertions: insertion_map[idx] = insertion_map.get(idx, "") + text
    result, last_idx = [], 0
    for idx in sorted(insertion_map.keys()):
        result.append(code[last_idx:idx]); result.append(insertion_map[idx]); last_idx = idx
    result.append(code[last_idx:]); return "".join(result)

# --- Transformation 2: Member Access Exchange (. <-> ->) ---

def apply_member_access_transform(code):
    token_pattern = re.compile(r'''
        (?P<INCLUDE>\#\s*include\s*<[^>]+>) | (?P<COMMENT_ML>/\*.*?\*/) | (?P<COMMENT_SL>//[^\n]*) |
        (?P<STRING>"(?:\\.|[^"\\])*") | (?P<CHAR>'(?:\\.|[^'\\])*') |
        (?P<NUMBER>(?:0[xX][0-9a-fA-F]+[uUlL]*) | (?:\d+\.\d*(?:[eE][+-]?\d+)?[fFlL]*) | (?:\.\d+(?:[eE][+-]?\d+)?[fFlL]*) | (?:\d+[eE][+-]?\d+[fFlL]*) | (?:\d+[uUlL]*)) |
        (?P<ID>[a-zA-Z_][a-zA-Z0-9_]*) | (?P<OP_ARROW>->) | (?P<OP_INC>\+\+) | (?P<OP_DEC>--) | (?P<OP_DOT>\.) |
        (?P<WS>\s+) | (?P<OTHER>.)
    ''', re.DOTALL | re.VERBOSE)
    tokens = []
    for match in token_pattern.finditer(code):
        for name, value in match.groupdict().items():
            if value is not None: tokens.append((name, value)); break
    idx = 0
    while idx < len(tokens):
        t_type, t_val = tokens[idx]
        if t_type in ('OP_ARROW', 'OP_DOT'):
            j, state = idx - 1, 0
            while j >= 0:
                ctype, cval = tokens[j]
                if ctype in ('WS', 'COMMENT_ML', 'COMMENT_SL'): j -= 1; continue
                if state == 0:
                    if ctype in ('OP_INC', 'OP_DEC'): j -= 1
                    elif cval in (')', ']'):
                        cp, op = cval, ('(' if cval == ')' else '[')
                        bal, j = 1, j - 1
                        while j >= 0 and bal > 0:
                            if tokens[j][1] == cp: bal += 1
                            elif tokens[j][1] == op: bal -= 1
                            j -= 1
                    elif ctype in ('ID', 'NUMBER', 'STRING', 'CHAR'): j -= 1; state = 1
                    else: break
                elif state == 1:
                    if ctype in ('OP_DOT', 'OP_ARROW'): j -= 1; state = 0
                    else: break
            start_idx = j + 1
            while start_idx < idx and tokens[start_idx][0] in ('WS', 'COMMENT_ML', 'COMMENT_SL'): start_idx += 1
            end_idx = idx + 1
            while end_idx < len(tokens) and tokens[end_idx][0] in ('WS', 'COMMENT_ML', 'COMMENT_SL'): end_idx += 1
            if start_idx < idx and end_idx < len(tokens) and tokens[end_idx][0] == 'ID':
                L_toks, R_toks = tokens[start_idx: idx], tokens[idx + 1: end_idx + 1]
                if t_type == 'OP_DOT':
                    new = [('OTHER', '('), ('OTHER', '&'), ('OTHER', '(')] + L_toks + [('OTHER', ')'), ('OTHER', ')'), ('OP_ARROW', '->')] + R_toks
                else:
                    new = [('OTHER', '('), ('OTHER', '*'), ('OTHER', '(')] + L_toks + [('OTHER', ')'), ('OTHER', ')'), ('OP_DOT', '.')] + R_toks
                tokens[start_idx: end_idx + 1] = new; idx = start_idx + len(new)
            else: idx += 1
        else: idx += 1
    return "".join(val for _, val in tokens)

# --- Transformation 3: Comma Expression Wrap ---

def wrap_expr(tokens):
    meaningful = [t for t in tokens if t[0] not in ('SPACE', 'COMMENT_SL', 'COMMENT_ML', 'PREPROC')]
    if not meaningful or any(t[0] == 'PREPROC' for t in tokens): return tokens
    if len(meaningful) == 1 and meaningful[0][0] == 'GROUP' and meaningful[0][1] == '{}': return tokens
    f = 0
    while f < len(tokens) and tokens[f][0] in ('SPACE', 'COMMENT_SL', 'COMMENT_ML'): f += 1
    l = len(tokens) - 1
    while l >= 0 and tokens[l][0] in ('SPACE', 'COMMENT_SL', 'COMMENT_ML'): l -= 1
    wrapper = [('PUNCT', '('), ('PUNCT', '('), ('ID', 'void'), ('PUNCT', ')'), ('NUMBER', '0'), ('PUNCT', ','), ('SPACE', ' ')]
    return tokens[:f] + wrapper + tokens[f:l + 1] + [('PUNCT', ')')] + tokens[l + 1:]

def transform_comma_wrap(ast_list, in_const=False, depth=0):
    EXCLUDE = {'if', 'for', 'while', 'switch', 'catch', 'sizeof', 'alignof', 'decltype', 'case'}
    TYPES = {'int', 'char', 'float', 'double', 'void', 'bool', 'auto', 'size_t'}
    i = 0
    while i < len(ast_list):
        node = ast_list[i]
        if node[0] == 'GROUP':
            ast_list[i] = ('GROUP', node[1], transform_comma_wrap(node[2], in_const or node[1] == '[]', depth + (1 if node[1] == '{}' else 0)))
            i += 1; continue
        if in_const: i += 1; continue
        if node[0] == 'ID' and node[1] == 'return':
            s = i + 1; e = s
            while e < len(ast_list) and not (ast_list[e][0] == 'PUNCT' and ast_list[e][1] == ';'): e += 1
            ast_list[s:e] = wrap_expr(ast_list[s:e]); i = e
        elif node[0] == 'OP_ASSIGN':
            if depth == 0: i += 1; continue
            s = i + 1; e = s
            while e < len(ast_list) and not (ast_list[e][0] == 'PUNCT' and ast_list[e][1] in (';', ',')): e += 1
            ast_list[s:e] = wrap_expr(ast_list[s:e]); i = e
        elif node[0] == 'ID' and node[1] not in EXCLUDE:
            nxt = i + 1
            while nxt < len(ast_list) and ast_list[nxt][0] in ('SPACE', 'COMMENT_ML', 'COMMENT_SL'): nxt += 1
            if nxt < len(ast_list) and ast_list[nxt][0] == 'GROUP' and ast_list[nxt][1] == '()':
                inner = ast_list[nxt][2]
                if not any(t[0] == 'ID' and t[1] in TYPES for t in inner):
                    new_inner, arg_s, j = [], 0, 0
                    while j <= len(inner):
                        if j == len(inner) or (inner[j][0] == 'PUNCT' and inner[j][1] == ','):
                            new_inner.extend(wrap_expr(inner[arg_s:j]))
                            if j < len(inner): new_inner.append(inner[j])
                            arg_s = j + 1
                        j += 1
                    ast_list[nxt] = ('GROUP', '()', new_inner)
                i = nxt + 1
            else: i += 1
        else: i += 1
    return ast_list

def apply_comma_wrap(code):
    tokens = []
    for m in token_pat.finditer(code): tokens.append((m.lastgroup, m.group()))
    ast = build_ast(tokens)
    return flatten_ast(transform_comma_wrap(ast))

# --- Transformation 4: Condition Wrap (你提供的新代码) ---

def apply_condition_wrap(code):
    n = len(code)
    insertions = []
    def is_id_char(c): return c.isalnum() or c == '_'
    def find_paren(start_idx):
        i = start_idx
        while i < n:
            if code[i].isspace(): i += 1
            elif code[i:i + 2] == '//':
                i += 2
                while i < n and code[i] != '\n': i += 1
                if i < n: i += 1
            elif code[i:i + 2] == '/*':
                i += 2
                while i < n and code[i:i+2] != '*/': i += 1
                if i < n: i += 2
            elif code[i] == '(': return i
            else: return -1
        return -1
    def find_matching_paren(start_idx):
        i = start_idx + 1; depth = 1
        while i < n:
            if code[i] == '"':
                i += 1
                while i < n:
                    if code[i] == '\\': i += 2
                    elif code[i] == '"': i += 1; break
                    else: i += 1
                continue
            if code[i] == "'":
                i += 1
                while i < n:
                    if code[i] == '\\': i += 2
                    elif code[i] == "'": i += 1; break
                    else: i += 1
                continue
            if code[i:i + 2] == '//':
                i += 2
                while i < n and code[i] != '\n': i += 1
                continue
            if code[i:i + 2] == '/*':
                i += 2
                while i < n and code[i:i+2] != '*/': i += 1
                if i < n: i += 2
                continue
            if code[i] == '(': depth += 1
            elif code[i] == ')':
                depth -= 1
                if depth == 0: return i
            i += 1
        return -1

    i = 0
    while i < n:
        if code[i] == '"':
            i += 1
            while i < n:
                if code[i] == '\\': i += 2
                elif code[i] == '"': i += 1; break
                else: i += 1
            continue
        if code[i] == "'":
            i += 1
            while i < n:
                if code[i] == '\\': i += 2
                elif code[i] == "'": i += 1; break
                else: i += 1
            continue
        if code[i:i + 2] == '//':
            i += 2
            while i < n and code[i] != '\n': i += 1
            continue
        if code[i:i + 2] == '/*':
            i += 2
            while i < n and code[i:i+2] != '*/': i += 1
            if i < n: i += 2
            continue

        is_word_start = (i == 0 or not is_id_char(code[i - 1]))
        if is_word_start:
            # if/while
            for kw in ['if', 'while']:
                kw_len = len(kw)
                if code[i:i + kw_len] == kw and (i + kw_len == n or not is_id_char(code[i + kw_len])):
                    p_start = find_paren(i + kw_len)
                    if p_start != -1:
                        p_end = find_matching_paren(p_start)
                        if p_end != -1:
                            insertions.append((p_start + 1, '('))
                            insertions.append((p_end, ')'))
            # for
            if code[i:i + 3] == 'for' and (i + 3 == n or not is_id_char(code[i + 3])):
                p_start = find_paren(i + 3)
                if p_start != -1:
                    p_end = find_matching_paren(p_start)
                    if p_end != -1:
                        j, d_p, d_b, d_br, s1, s2 = p_start + 1, 0, 0, 0, -1, -1
                        while j < p_end:
                            if code[j] == '"':
                                j += 1
                                while j < p_end:
                                    if code[j] == '\\': j += 2
                                    elif code[j] == '"': j += 1; break
                                    else: j += 1
                                continue
                            if code[j] == "'":
                                j += 1
                                while j < p_end:
                                    if code[j] == '\\': j += 2
                                    elif code[j] == "'": j += 1; break
                                    else: j += 1
                                continue
                            if code[j] == '(': d_p += 1
                            elif code[j] == ')': d_p -= 1
                            elif code[j] == '{': d_b += 1
                            elif code[j] == '}': d_b -= 1
                            elif code[j] == '[': d_br += 1
                            elif code[j] == ']': d_br -= 1
                            elif code[j] == ';' and d_p == 0 and d_b == 0 and d_br == 0:
                                if s1 == -1: s1 = j
                                elif s2 == -1: s2 = j
                            j += 1
                        if s1 != -1 and s2 != -1:
                            if code[s1+1:s2].strip():
                                insertions.append((s1 + 1, '('))
                                insertions.append((s2, ')'))
        i += 1
    insertions.sort(key=lambda x: x[0])  # 从前往后排

    parts = []
    last_idx = 0
    for idx, char in insertions:
        parts.append(code[last_idx:idx])
        parts.append(char)
        last_idx = idx
    parts.append(code[last_idx:])

    return "".join(parts)

# --- 其他常规转换 ---

def apply_splitting(code):
    pat = re.compile(r'^(\s*)([a-zA-Z_]\w*\s*\*?\s+)([a-zA-Z_]\w*)\s*=\s*([^;]+);', re.MULTILINE)

    def repl(m):
        indent, t_str, v_name, expr = m.group(1), m.group(2).strip(), m.group(3).strip(), m.group(4).strip()
        if "const " in t_str or "return" in t_str: return m.group(0)
        return f"{indent}{t_str} {v_name};\n{indent}{v_name} = {expr};"

    return pat.sub(repl, code)


def apply_extraction(code):
    lines = code.split('\n')
    new_lines = []
    counter = 0

    for line in lines:
        if len(line) > 500:
            new_lines.append(line)
            continue

        if "=" in line and not any(line.strip().startswith(s) for s in ("if", "for", "while")):
            eq_idx = line.find('=')
            semi_idx = line.find(';', eq_idx)
            # 只有在等号后能找到分号时才处理
            if semi_idx != -1:
                expr = line[eq_idx + 1:semi_idx]
                m = re.search(r'\(([^()]+)\)', expr)
                if m:
                    inner = m.group(1)
                    if ',' not in inner and not inner.strip().endswith(('*', '&')) and 'sizeof' not in line:
                        if any(op in inner.replace('->', '') for op in ['+', '-', '*', '/', '<<', '>>', '|', '&', '^']):
                            tmp = f"__tmp_{counter}";
                            counter += 1
                            indent_m = re.match(r'^\s*', line)
                            indent = indent_m.group(0) if indent_m else ""
                            new_lines.append(f"{indent}auto {tmp} = {inner};")
                            line = line.replace(f"({inner})", f"({tmp})")

        new_lines.append(line)
    return '\n'.join(new_lines)

def apply_inversion(code):
    def find_match(txt, s, o, c):
        cnt = 0
        for idx in range(s, len(txt)):
            if txt[idx] == o: cnt += 1
            elif txt[idx] == c:
                cnt -= 1
                if cnt == 0: return idx
        return -1
    out, i, n = [], 0, len(code)
    while i < n:
        if code[i:i + 2] == "if" and (i == 0 or not code[i - 1].isalnum()) and "else " not in code[max(0, i - 5):i]:
            sc = code.find('(', i)
            if sc != -1:
                ec = find_match(code, sc, '(', ')')
                if ec != -1:
                    cond = code[sc + 1:ec]
                    st = code.find('{', ec)
                    if st != -1 and code[ec + 1:st].strip() == "":
                        et = find_match(code, st, '{', '}')
                        if et != -1:
                            true_b = code[st + 1:et]; el_idx = code.find('else', et)
                            if el_idx != -1 and code[et + 1:el_idx].strip() == "":
                                sf = code.find('{', el_idx)
                                if sf != -1:
                                    ef = find_match(code, sf, '{', '}')
                                    if ef != -1:
                                        out.append(f"if (!({cond})) {{\n{code[sf + 1:ef]}\n}} else {{\n{true_b}\n}}")
                                        i = ef + 1; continue
                            out.append(f"if (!({cond})) {{ /* empty */ }} else {{\n{true_b}\n}}")
                            i = et + 1; continue
        out.append(code[i]); i += 1
    return "".join(out)

def apply_boolean_expansion(source):
    out, i, n = [], 0, len(source)
    while i < n:
        if source[i:i + 2] in ('//', '/*'):
            end = source.find('\n' if source[i + 1] == '/' else '*/', i + 2)
            step = 1 if source[i + 1] == '/' else 2
            curr_end = (end + step) if end != -1 else n
            out.append(source[i:curr_end]); i = curr_end; continue
        if source[i] in '"\'':
            q = source[i]; out.append(q); i += 1
            while i < n and source[i] != q:
                if source[i] == '\\': out.append(source[i]); i += 1
                out.append(source[i]); i += 1
            if i < n: out.append(q); i += 1
            continue
        if source[i].isalpha() or source[i] == '_':
            s = i
            while i < n and (source[i].isalnum() or source[i] == '_'): i += 1
            word = source[s:i]
            if word in ('if', 'while'):
                ti = i
                while ti < n and source[ti].isspace(): ti += 1
                if ti < n and source[ti] == '(':
                    out.extend([word, source[i:ti], '(']); ti += 1
                    pc, expr = 1, []
                    while ti < n and pc > 0:
                        if source[ti] == '(': pc += 1
                        elif source[ti] == ')': pc -= 1
                        if pc > 0: expr.append(source[ti])
                        ti += 1
                    e_str = "".join(expr)
                    out.append(f"({e_str}) && 1" if ';' not in e_str else e_str)
                    out.append(')'); i = ti; continue
            out.append(word); continue
        out.append(source[i]); i += 1
    return "".join(out)

# --- 混淆调度 ---

def obfuscate(code):
    # 增加硬性熔断：超过 8000 字符的怪物级代码直接返回，不参与混淆
    # if len(code) > 8000:
    #     return code

    transformations = [
        apply_splitting,
        apply_extraction,
        apply_inversion,
        apply_boolean_expansion,
        apply_comma_wrap,
        apply_member_access_transform,
        apply_else_padding,
        apply_condition_wrap
    ]

    # 每次随机抽 2 到 4 个变换进行组合（避免套娃太多层导致语法彻底崩溃）
    num = random.randint(3, min(5, len(transformations)))
    selected_indices = sorted(random.sample(range(len(transformations)), num))

    for idx in selected_indices:
        try:
            code = transformations[idx](code)
        except Exception as e:
            continue

    return code

if __name__ == "__main__":
    src = sys.stdin.read()
    if src:
        sys.stdout.write(obfuscate(src))