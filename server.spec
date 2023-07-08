# server.spec

block_cipher = None

a = Analysis(['server.py'],
             pathex=['.'],
             binaries=[],
             datas=[('artroom_helpers', 'artroom_helpers'),
                    ('backend', 'backend'),
                    ('gfpgan', 'gfpgan'),
                    ('model_merger.py', '.'),
                    ('safe.py', '.'),
                    ('stable_diffusion.py', '.'),
                    ('tags_pull.py', '.'),
                    ('upscale.py', '.')],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)

pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)

exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='artroom',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True,
	  icon='artroom_icon.ico'
)