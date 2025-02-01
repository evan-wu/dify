"""merge heads

Revision ID: 5977a0565e82
Revises: b90b587766cb, a91b476a53de
Create Date: 2025-02-01 12:46:19.914739

"""
from alembic import op
import models as models
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '5977a0565e82'
down_revision = ('b90b587766cb', 'a91b476a53de')
branch_labels = None
depends_on = None


def upgrade():
    pass


def downgrade():
    pass
