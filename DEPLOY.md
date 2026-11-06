# Deploying to Hugging Face Spaces (free)

The app runs as a Docker Space on the free **CPU basic** hardware
(2 vCPU, 16 GB RAM). `deploy_hf.py` uploads only the files the server needs.

## One-time setup

1. Create a free account at https://huggingface.co/join.
2. Create a **write** token: https://huggingface.co/settings/tokens → *Create new token* → type **Write**.
3. Log in from the project folder (paste the token when asked; answer `n` to the git-credential question):

   ```powershell
   tts-env\Scripts\hf auth login
   ```

## Deploy (and redeploy after every change)

```powershell
tts-env\Scripts\python deploy_hf.py <your-hf-username>/hindi-register-translation
```

Add `--private` on the first run to make the Space visible only to you.
Add `--dry-run` to see which files would be uploaded without uploading.

The first build takes about 10–15 minutes (torch + model downloads). Watch it in
the **Logs** tab of `https://huggingface.co/spaces/<user>/hindi-register-translation`.
The status changes from *Building* to *Running* when it's ready.

## Secrets (Space → Settings → Variables and secrets → New secret)

| Name | Value | Needed? |
|---|---|---|
| `GNANI_API_KEY` | your Gnani key | yes, for Timbre TTS / Prisma ASR |
| `APP_USERNAME` | any username | recommended on a public Space |
| `APP_PASSWORD` | any password | recommended on a public Space |

Setting both `APP_USERNAME` and `APP_PASSWORD` adds a login screen so strangers can't
spend your Gnani credits. The Space restarts automatically after you add secrets.
Never upload `.env`; the deploy script already skips it.

## Notes

- Free Spaces sleep after 48 h without visitors. The first visit afterwards takes a minute or two to wake up.
- Parler TTS (~3.5 GB) downloads the first time someone picks it, so that first request is slow. Timbre is the default and fast.
- If the build fails, the error is at the bottom of the Logs tab.
