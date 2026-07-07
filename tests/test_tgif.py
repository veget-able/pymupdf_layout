import pymupdf

if pymupdf.mupdf_version_tuple >= (1, 28):
    import pymupdf.tgif
    from pymupdf import tgif

    import os


    def FzStextGridPositions_str(stext_grid_positions, indent=0):
        '''
        Returns multi-line string description of a FzStextGridPositions.
        '''
        if isinstance(indent, int):
            indent = ' ' * indent
        ret = ''
        ret += f'{indent}max_uncertainty={stext_grid_positions.m_internal.max_uncertainty}\n'
        ret += f'{indent}len={stext_grid_positions.m_internal.len}:\n'
        for i in range(stext_grid_positions.m_internal.len):
            ret += f'{indent}    {stext_grid_positions.list(i)}\n'
        return ret

    def test_tgif():
        path = os.path.normpath(f'{__file__}/../../tests/test_activate.pdf')
        with pymupdf.open(path) as document:
            for page in document:
                print(f'{page=}', flush=1)
                print(f'{page.this=}', flush=1)
                bound = pymupdf.mupdf.fz_bound_page(page.this)
                print(f'{bound=}', flush=1)
                r, xpos, ypos = tgif.fz_visual_table_grid_finder(page, bound)
                print(f'{r=}', flush=1)
                #print(f'{xpos=}', flush=1)
                #print(f'{ypos=}', flush=1)
                print(f'xpos:\n{FzStextGridPositions_str(xpos, 4)}')
                print(f'ypos:\n{FzStextGridPositions_str(ypos, 4)}')


    def test_tgif2():
        path = os.path.normpath(f'{__file__}/../../tests/test_activate.pdf')
        with pymupdf.open(path) as document:
            for page in document:
                print(f'{page=}', flush=1)
                print(f'{page.this=}', flush=1)
                textpage = page.get_textpage()
                bound = pymupdf.mupdf.fz_bound_page(page.this)
                stextblock = tgif.fz_find_visual_table_within_bounds(textpage, page, bound)
                print(f'{stextblock=}', flush=1)

else:
    try:
        from pymupdf import tgif
    except ImportError:
        pass
    else:
        assert 0, 'Expected `from pymupdf import tgif` to fail.'
