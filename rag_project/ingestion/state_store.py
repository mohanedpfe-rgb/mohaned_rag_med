        if document_id is None or worker_id is None:
            return False
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                with self._connect() as connection:
                    cursor = connection.execute("UPDATE documents SET lease_owner = NULL, lease_expires_at = NULL, heartbeat_at = NULL, modified_at = ? WHERE document_id = ? AND lease_owner = ?", (utc_now(), document_id, worker_id))
                return cursor.rowcount == 1
            except Exception as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(0.01 * (attempt + 1))
        # Lease cleanup is deliberately non-throwing. The ingestion caller is
        # already at a cleanup boundary; leaking the storage exception here can
        # incorrectly turn an otherwise published SUCCESS into a failure.
        return False
