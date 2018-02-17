seq2seq-attn Shared Information and Network Dissection Scripts
==============================================================

To use these scripts you will first need to generate description files using `describe.lua` from here: [https://github.com/dabbler0/nmt-shared-information]().

More information on usage in the leading comment of given files.

The visualization part of the server should work, but the modification server currently will not work because of how much it depended on my directory structure. The modification part of the server will also depend on an altered version of seq2seq-attn available here: [https://github.com/dabbler0/seq2seq-attn-modify]().

This repository does not currently contain the "masking-out" procedure but should soon. The masking-out procedure will also depend on this altered version of seq2seq-attn.
