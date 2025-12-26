# LDA_PP
## セットアップ
    conda activate (環境名)
    cd desktop/LDA_PP/program
    python setup.py build_ext --inplace
    cd coord_conv
    sh setup.sh

# トラブルシューティング
## git pullし忘れてcommitしてしまった場合
    git pull --rebase
    git push