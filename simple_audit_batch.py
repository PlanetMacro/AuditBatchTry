#!/usr/bin/env python3
import base64
import getpass
import json
import os
import sys
from dataclasses import dataclass
from typing import Tuple

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from cryptography.hazmat.backends import default_backend
from cryptography.exceptions import InvalidTag
import secrets

FILENAME = "OPENAI.API_KEY"

# ---- Configuration (tune if needed) ----
SCRYPT_N = 2**14  # CPU/memory cost
SCRYPT_R = 8
SCRYPT_P = 1
KEY_LEN = 32      # 256-bit AES key
SALT_LEN = 16
NONCE_LEN = 12    # AES-GCM standard nonce size

@dataclass
class EncryptedBlob:
    version: int
    salt_b64: str
    nonce_b64: str
    ciphertext_b64: str
    kdf: str = "scrypt"
    n: int = SCRYPT_N
    r: int = SCRYPT_R
    p: int = SCRYPT_P

def _derive_key(password: str, salt: bytes, n: int, r: int, p: int) -> bytes:
    kdf = Scrypt(
        salt=salt, length=KEY_LEN, n=n, r=r, p=p, backend=default_backend()
    )
    return kdf.derive(password.encode("utf-8"))

def _encrypt(api_key: str, password: str) -> Tuple[EncryptedBlob, bytes, bytes]:
    salt = secrets.token_bytes(SALT_LEN)
    key = _derive_key(password, salt, SCRYPT_N, SCRYPT_R, SCRYPT_P)
    nonce = secrets.token_bytes(NONCE_LEN)
    aesgcm = AESGCM(key)
    ciphertext = aesgcm.encrypt(nonce, api_key.encode("utf-8"), associated_data=None)
    blob = EncryptedBlob(
        version=1,
        salt_b64=base64.b64encode(salt).decode("ascii"),
        nonce_b64=base64.b64encode(nonce).decode("ascii"),
        ciphertext_b64=base64.b64encode(ciphertext).decode("ascii"),
    )
    return blob, salt, nonce

def _decrypt(blob: EncryptedBlob, password: str) -> str:
    if blob.kdf != "scrypt":
        raise ValueError("Unsupported KDF.")
    salt = base64.b64decode(blob.salt_b64)
    nonce = base64.b64decode(blob.nonce_b64)
    ciphertext = base64.b64decode(blob.ciphertext_b64)

    key = _derive_key(password, salt, blob.n, blob.r, blob.p)
    aesgcm = AESGCM(key)
    plaintext = aesgcm.decrypt(nonce, ciphertext, associated_data=None)
    return plaintext.decode("utf-8")

def _write_secure_file(path: str, content: str) -> None:
    # Write atomically to avoid partial files
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(content)
    # Restrict permissions on POSIX systems
    try:
        os.chmod(tmp_path, 0o600)
    except Exception:
        pass
    os.replace(tmp_path, path)

def create_api_key_file() -> str:
    print("No API key file found. Generating a new encrypted key store.")
    api_key = getpass.getpass("Enter OpenAI API key (input hidden): ").strip()
    if not api_key:
        print("Empty API key; abort.", file=sys.stderr)
        sys.exit(1)

    # Prompt for password twice to reduce typos
    pw1 = getpass.getpass("Create a password to encrypt the key: ")
    pw2 = getpass.getpass("Re-enter the password: ")
    if pw1 != pw2:
        print("Passwords do not match; abort.", file=sys.stderr)
        sys.exit(1)

    blob, _salt, _nonce = _encrypt(api_key, pw1)
    json_text = json.dumps(blob.__dict__, indent=2)
    _write_secure_file(FILENAME, json_text)
    print(f"Encrypted key saved to ./{FILENAME}")
    return api_key

def load_api_key_file() -> str:
    with open(FILENAME, "r", encoding="utf-8") as f:
        data = json.load(f)
    try:
        blob = EncryptedBlob(**data)
    except TypeError as e:
        raise ValueError("Corrupt or unsupported file format.") from e

    password = getpass.getpass("Enter password to decrypt API key: ")
    try:
        api_key = _decrypt(blob, password)
    except InvalidTag:
        # Wrong password or tampered file
        raise ValueError("Decryption failed: incorrect password or file has been modified.")
    return api_key

def main() -> None:
    """
    - If OPENAI.API_KEY doesn't exist: prompt for API key and password, encrypt, and save.
    - If it exists: prompt for password, decrypt, and load the API key into `api_key`.
    """
    if not os.path.exists(FILENAME):
        api_key = create_api_key_file()
    else:
        print(f"Existing key store found at ./{FILENAME}")
        try:
            api_key = load_api_key_file()
        except ValueError as e:
            print(str(e), file=sys.stderr)
            sys.exit(1)

    # The decrypted API key is now available in this variable:
    # Use it as needed (do not print it).
    # For demonstration, just confirm success without revealing secrets.
    # e.g., set environment variable if you wish:
    # os.environ["OPENAI_API_KEY"] = api_key
    print("API key loaded into variable `api_key`.")

if __name__ == "__main__":
    main()

